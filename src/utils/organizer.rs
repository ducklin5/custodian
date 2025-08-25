use dioxus::prelude::*;
use dioxus::signals::{Signal, SyncStorage};

// use itertools::Itertools; // avoid Iterator adapters that deepen type recursion

use crate::utils::AsyncPtrProp;

const FETCH_BATCH_SIZE: u32 = 200;

#[derive(Clone, Debug)]
pub struct MessageInfo {
    pub(crate) uid: u32,
    pub(crate) subject: String,
    pub(crate) date: String,
    pub(crate) sender: String,
    pub(crate) sender_name: String,
    #[allow(dead_code)]
    pub(crate) body_text: String,
}

pub fn move_messages_to_trash(
    session: crate::utils::AsyncPtrProp<imap::Session<native_tls::TlsStream<std::net::TcpStream>>>,
    mailbox: String,
    msg_uids: Vec<u32>,
) -> Result<Vec<u32>, String> {
    let seq = msg_uids
        .iter()
        .map(|uid| uid.to_string())
        .collect::<Vec<_>>()
        .join(",");
    match session.lock() {
        Err(e) => Err(format!("Failed to lock session: {}", e)),
        Ok(mut session) => {
            // Best-effort keepalive and mailbox selection to avoid stale state
            let _ = (&mut *session).noop();
            let _ = (&mut *session).select(&mailbox);
            let mut success_uids = Vec::new();
            let mut failed_uids = Vec::new();
            session.debug = true;
            let mut last_err: Option<String> = None;
            let mut attempt = 0;
            let max_attempts = 2;
            let mut moved_ok = false;
            while attempt < max_attempts {
                match session.uid_mv(&seq, "Custodian/Trash") {
                    Ok(_) => {
                        moved_ok = true;
                        success_uids.extend(msg_uids.clone());
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e.to_string());
                        // Brief pause and NOOP to resync TLS/session
                        let _ = (&mut *session).noop();
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        attempt += 1;
                    }
                }
            }
            if moved_ok {
                println!("Successfully moved {} messages to trash", msg_uids.len());
                Ok(success_uids)
            } else {
                println!(
                    "Batch move failed: {:?}. Trying individual moves...",
                    last_err
                );
                let _ = (&mut *session).noop();
                for &uid in &msg_uids {
                    let uid_s = uid.to_string();
                    match (&mut *session).uid_mv(&uid_s, "Custodian/Trash") {
                        Ok(_) => {
                            success_uids.push(uid);
                            println!("Moved message {} to trash", uid);
                        }
                        Err(e) => {
                            println!("MOVE failed for {}: {}.", uid, e);
                            failed_uids.push(uid);
                        }
                    }
                }
                if !failed_uids.is_empty() {
                    println!(
                        "Failed to move {} messages: {:?}",
                        failed_uids.len(),
                        failed_uids
                    );
                    Ok(success_uids)
                } else {
                    Err(format!("Failed to move messages: {:?}", failed_uids))
                }
            }
        }
    }
}

// Function to fetch messages from IMAP server
pub fn fetch_messages(
    session: AsyncPtrProp<imap::Session<native_tls::TlsStream<std::net::TcpStream>>>,
    mailbox: String,
    mut server_msg_count: Signal<u32, SyncStorage>,
    mut msg_signal: Signal<Vec<MessageInfo>, SyncStorage>,
    with_body: bool,
) -> Result<Vec<MessageInfo>, String> {
    if let Ok(mut session_guard) = session.lock() {
        // First, select the mailbox
        println!("Selecting mailbox: {}", mailbox);
        if let Err(e) = (&mut *session_guard).select(&mailbox) {
            return Err(format!("Failed to select mailbox: {}", e));
        }

        // Now check the mailbox status after selection
        println!("Checking mailbox status...");
        match (&mut *session_guard).status(&mailbox, "(MESSAGES UIDNEXT UIDVALIDITY)") {
            Ok(status) => {
                println!("Mailbox status: {:?}", status);
                if status.exists == 0 {
                    println!(
                        "Warning: Mailbox shows 0 messages, but this might be due to Yahoo's limited view mode"
                    );
                }
            }
            Err(e) => {
                println!("Warning: Failed to get mailbox status: {}", e);
            }
        };

        // Try to get all UIDs - this might be limited by Yahoo's view mode
        println!("Searching for all UIDs...");
        let uid_vec = match (&mut *session_guard).uid_search("1:*") {
            Ok(uids) => {
                let uid_list: Vec<u32> = uids.into_iter().collect();
                println!("Found {} UIDs", uid_list.len());
                server_msg_count.set(uid_list.len() as u32);
                if uid_list.is_empty() {
                    println!("No UIDs found - trying alternative approach...");
                    // Try a different approach for Yahoo's limited view
                    match (&mut *session_guard).uid_search("ALL") {
                        Ok(alt_uids) => {
                            let alt_list: Vec<u32> = alt_uids.into_iter().collect();
                            println!("Alternative search found {} UIDs", alt_list.len());
                            server_msg_count.set(alt_list.len() as u32);
                            alt_list
                        }
                        Err(e) => {
                            return Err(format!("Failed to find any UIDs: {}", e));
                        }
                    }
                } else {
                    uid_list
                }
            }
            Err(e) => {
                return Err(format!("Failed to search for UIDs: {}", e));
            }
        };

        // Fetch messages with pagination
        let mut all_message_infos: Vec<MessageInfo> = Vec::new();

        for batch in uid_vec.chunks(FETCH_BATCH_SIZE as usize) {
            let sequence_set = batch
                .iter()
                .map(|uid| uid.to_string())
                .collect::<Vec<_>>()
                .join(",");
            println!("Fetching UID batch: {}...", sequence_set);

            let fetch_str = if with_body {
                "(ENVELOPE BODY[TEXT])"
            } else {
                "ENVELOPE"
            };
            let messages = match (&mut *session_guard).uid_fetch(&sequence_set, fetch_str) {
                Ok(messages) => messages,
                Err(e) => {
                    return Err(format!("Failed to fetch messages: {}", e));
                }
            };

            for message in messages.iter() {
                let uid = message.uid.unwrap_or(0);
                if uid == 0 {
                    return Err(format!("Invalid UID: {}", uid));
                }

                // Extract subject from envelope
                let subject = message
                    .envelope()
                    .and_then(|env| env.subject.as_ref())
                    .map(|subj| {
                        // Decode RFC 2047 encoded words (like =?UTF-8?Q?...?=)
                        let subject_str = String::from_utf8_lossy(subj);
                        match rfc2047_decoder::decode(subject_str.as_bytes()) {
                            Ok(decoded) => decoded,
                            Err(_) => subject_str.to_string(), // Fallback to original if decoding fails
                        }
                    })
                    .unwrap_or_else(|| "No Subject".to_string());

                // Extract date from envelope
                let date = message
                    .envelope()
                    .and_then(|env| env.date.as_ref())
                    .map(|date| String::from_utf8_lossy(date).to_string())
                    .unwrap_or_else(|| "No Date".to_string());

                let (sender, sender_name) = message
                    .envelope()
                    .and_then(|env| env.from.as_ref())
                    .and_then(|from| from.first())
                    .map(|addr| {
                        let name = addr
                            .name
                            .map(|n| String::from_utf8_lossy(n))
                            .unwrap_or("Unknown".into());
                        let mailbox = addr
                            .mailbox
                            .map(|m| String::from_utf8_lossy(m))
                            .unwrap_or_default();
                        let host = addr
                            .host
                            .map(|h| String::from_utf8_lossy(h))
                            .unwrap_or_default();

                        // Use display name if available, otherwise use email address
                        (
                            format!("{}@{}", mailbox, host)
                                .to_lowercase()
                                .trim()
                                .to_string(),
                            name.into(),
                        )
                    })
                    .unwrap_or(("Unknown".to_string(), "Unknown".to_string()));

                let body_text = message
                    .text()
                    .map(|t| String::from_utf8_lossy(t).to_string())
                    .unwrap_or("No Body".to_string());

                all_message_infos.push(MessageInfo {
                    uid,
                    subject,
                    date,
                    sender,
                    sender_name,
                    body_text,
                });
            }
            if let Ok(mut msg_signal_guard) = msg_signal.try_write() {
                *msg_signal_guard = all_message_infos.clone();
            }
        }

        // Sort by UID (newest first)
        all_message_infos.sort_by(|a, b| b.uid.cmp(&a.uid));
        Ok(all_message_infos)
    } else {
        Err("Failed to lock session".to_string())
    }
}

// Fetch the full message body (text) for a specific UID from a mailbox
pub fn fetch_message_body(
    session: AsyncPtrProp<imap::Session<native_tls::TlsStream<std::net::TcpStream>>>,
    mailbox: &str,
    uid: u32,
) -> Result<String, String> {
    if let Ok(mut session_guard) = session.lock() {
        // Assume mailbox has already been selected by caller; avoid re-select churn
        // Keepalive to avoid idle disconnects
        let _ = (&mut *session_guard).noop();

        // Fetch body text for the specific UID
        let seq = uid.to_string();
        let fetch_items = "(BODY[TEXT])";
        let messages = (&mut *session_guard)
            .uid_fetch(&seq, fetch_items)
            .map_err(|e| format!("Failed to fetch body for UID {}: {}", uid, e))?;

        if let Some(message) = messages.iter().next() {
            let body_text = message
                .text()
                .map(|t| String::from_utf8_lossy(t).to_string())
                .unwrap_or_else(|| String::new());
            Ok(body_text)
        } else {
            Err(format!("No message found for UID {}", uid))
        }
    } else {
        Err("Failed to lock session".to_string())
    }
}
