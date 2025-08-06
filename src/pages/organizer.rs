use dioxus::prelude::*;

use burn::backend::wgpu::WgpuDevice;

use itertools::Itertools;

use std::collections::HashMap;

use crate::utils::AsyncPtrProp;
use crate::ai::models::llama::HFLlama;
use crate::ai::prompts::email_classification_prompt;

const FETCH_BATCH_SIZE: u32 = 200;

#[derive(Clone, Props, PartialEq)]
pub struct OrganizerPageProps {
    session: AsyncPtrProp<imap::Session<native_tls::TlsStream<std::net::TcpStream>>>,
    email: String,
    mailbox: String,
    on_back: EventHandler<(
        AsyncPtrProp<imap::Session<native_tls::TlsStream<std::net::TcpStream>>>,
        String,
        String,
    )>,
}

#[derive(Clone, Debug)]
struct MessageInfo {
    uid: u32,
    subject: String,
    date: String,
    sender: String,
    sender_name: String,
    body_text: String,
}

// Function to fetch messages from IMAP server
fn fetch_messages(
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
            let sequence_set = batch.iter().map(|uid| format!("{}", uid)).join(",");
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

                let body_text = message.text()
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

#[component]
pub fn OrganizerPage(props: OrganizerPageProps) -> Element {
    let OrganizerPageProps {
        session,
        email,
        mailbox,
        on_back,
    } = props;

    let mut msg_fetch_done = use_signal(|| false);
    let mut error = use_signal(|| Option::<String>::None);
    let mut messages_signal = use_signal_sync(|| Vec::<MessageInfo>::new());
    let server_msg_count = use_signal_sync(|| 0);
    let mut smart_progress = use_signal_sync(|| 0);
    let mut smart_complete = use_signal_sync(|| false);
    let session_signal = use_signal_sync(|| session.clone());
    let mailbox_signal = use_signal_sync(|| mailbox.clone());
    let mut group_by_sender = use_signal(|| false);
    let mut group_by_category_active = use_signal(|| false);

    let mut messages_by_sender = use_signal_sync(|| HashMap::<String, Vec<MessageInfo>>::new());
    let messages_by_category = use_signal_sync(|| HashMap::<String, Vec<MessageInfo>>::new());
    
    // Generator loading state
    let mut generator_loading = use_signal(|| true);
    let mut generator = use_signal_sync(|| Option::<AsyncPtrProp<HFLlama>>::None);


    // Fetch messages when component mounts
    use_effect(move || {
        spawn(async move {
            let result =
                tokio::task::spawn_blocking(move || fetch_messages(session_signal().clone(), mailbox_signal(), server_msg_count.clone(), messages_signal.clone(), false))
                    .await
                    .unwrap();
            match result {
                Ok(msg_infos) => {
                    messages_signal.set(msg_infos);
                }
                Err(e) => {
                    error.set(Some(e));
                }
            }
            msg_fetch_done.set(true);
        });
    });

    
    // Initialize model in use_effect
    use_effect(move || {
        spawn(async move {
            let device = WgpuDevice::default();
            let model_id = "HuggingFaceTB/SmolLM2-135M";
            let revision = "main";
            
            println!("Loading model for generate test...");
            let result = tokio::task::spawn_blocking(move || {
                let mut model = HFLlama::new(model_id, revision, device).expect("Failed to load Llama model");
                model.add_terminator("<|end:categories|>");
                model.add_terminator("<");
                model.add_terminator("\n");
                model.add_terminator("Task");
                AsyncPtrProp::new(model)
            }).await;
            
            match result {
                Ok(model) => {
                    generator.set(Some(model));
                    generator_loading.set(false);
                }
                Err(e) => {
                    println!("Failed to load model: {:?}", e);
                    generator_loading.set(false);
                }
            }
        });
    });

    use_effect(move || {
        if group_by_sender() {
            let map: HashMap<String, Vec<MessageInfo>> = messages_signal
                .read()
                .iter()
                .into_grouping_map_by(|m| m.sender.clone())
                .aggregate(|acc, _key, msg| {
                    let mut group = acc.unwrap_or(vec![]);
                    group.push(msg.clone());
                    Some(group)
                });
            messages_by_sender.set(map);
        }
    });

    let group_by_category_callback = use_callback(move |_| {
        if !generator_loading() && generator().is_some() {
            group_by_category_active.set(true);
            let messages_clone = messages_signal.clone();
            let mut messages_by_category_clone = messages_by_category.clone();
            let generator_clone = generator.clone();
            let _result = tokio::task::spawn_blocking(move || {
                let mut progress = 0;
                for msg in messages_clone.read().iter() {
                    let mut groups = messages_by_category_clone.write();
                    let top_5_categories = groups.iter().sorted_by(|a, b| b.1.len().cmp(&a.1.len())).take(5).map(|k| k.0.clone()).collect::<Vec<_>>();
                    let prompt = email_classification_prompt(top_5_categories, &msg.subject, &msg.sender);
                    if let Some(model) = generator_clone() {
                        let result = model.lock().unwrap().generate(prompt, None, None).unwrap_or("Err".into());
                        let labels = result
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .unique()
                            .filter(|s| !s.is_empty())
                            .collect::<Vec<_>>();
                        println!("Labels: {:?}", labels);
                        for label in labels {
                            groups.entry(label).or_insert(Vec::new()).push(msg.clone());
                        }
                    }
                    progress += 1;
                    *smart_progress.write() = progress;
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                *smart_complete.write() = true;
            });
        }
    });

    let local_msg_count = messages_signal.read().len();
    let sorted_msg_by_sender = {
        let msg_by_sender = messages_by_sender.read();
        msg_by_sender
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .sorted_by(|a, b| b.1.len().cmp(&a.1.len()))
        .collect::<Vec<_>>()
    };

    
    let sorted_msg_by_category = {
        let msg_by_category = messages_by_category.read();
        msg_by_category
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .sorted_by(|a, b| b.1.len().cmp(&a.1.len()))
            .collect::<Vec<_>>()
    };

    rsx! {
        div { class: "min-h-screen flex flex-col items-center justify-center bg-slate-500 gap-6 p-20",
            div { class: "bg-white p-8 rounded-2xl shadow-lg min-w-[350px] w-full m-10 text-center",
                h1 { class: "text-2xl font-bold text-blue-700 mb-4", "Mailbox Organizer" }
                div { class: "mb-4",
                    p { class: "text-gray-600", "Email: {email}" }
                    p { class: "text-gray-600", "Mailbox: {mailbox_signal()}" }
                }

                if !msg_fetch_done() {
                    div { class: "mb-6",
                        p { class: "text-gray-600", "Loading messages..." }
                        p { class: "text-gray-600", "Messages fetched: {local_msg_count} / {server_msg_count()}" }
                    }
                }  else if let Some(err) = error() {
                    div { class: "mb-6",
                        p { class: "text-red-600", "Error: {err}" }
                    }
                } else {
                    div { class: "mb-6",
                        p { class: "text-gray-600 mb-4", "Found {local_msg_count} messages" }
                        div { class: "max-h-96 overflow-y-auto border-spacing-0",
                            style: "box-shadow: inset 10px -8px 8px -8px rgba(0, 0, 0, 0.5), inset -10px 0 8px 8px rgba(0, 0, 0, 0.5);",
                            table { class: "w-full border-separate border-spacing-0",
                                thead {
                                    tr { class: "bg-gray-100 sticky top-0 bg-gray-100",
                                        style: "box-shadow: 0 3px 8px 0px rgba(0, 0, 0, 0.5);",
                                        th { class: "border-3 border-black px-4 py-2 text-left ", "UID" }
                                        th { class: "border-3 border-black px-4 py-2 text-left ", "Subject" }
                                        th { class: "border-3 border-black px-4 py-2 text-left ", "Sender" }
                                        th { class: "border-3 border-black px-4 py-2 text-left ", "Date" }
                                    }
                                }
                                tbody {
                                    for message in messages_signal().iter() {
                                        tr { class: "hover:bg-gray-50",
                                            td { class: "border border-gray-300 px-4 py-2", "{message.uid}" }
                                            td { class: "border border-gray-300 px-4 py-2 text-left", "{message.subject}" }
                                            td { class: "border border-gray-300 px-4 py-2", "[{message.sender}] {message.sender_name}" }
                                            td { class: "border border-gray-300 px-4 py-2", "{message.date}" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                div { class: "flex flex-row gap-4",
                    button {
                        class: "w-full py-3 bg-gray-300 text-gray-800 rounded-lg font-bold text-lg hover:bg-gray-400 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed", 
                        onclick: move |_| {
                            on_back.call((session_signal().clone(), email.clone(), mailbox_signal().clone()));
                        },
                        "Back to Home"
                    }
                    button {
                        class: "w-full py-3 bg-gray-300 text-gray-800 rounded-lg font-bold text-lg hover:bg-gray-400 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed",
                        disabled: !msg_fetch_done(),
                        onclick: move |_| {
                            group_by_sender.set(true);
                        },
                        "Group by Sender"
                    }
                    button {
                        class: "w-full py-3 bg-gray-300 text-gray-800 rounded-lg font-bold text-lg hover:bg-gray-400 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed",
                        disabled: !msg_fetch_done() || generator_loading(),
                        onclick: move |_| {
                            group_by_category_callback.call(());
                        },
                        if generator_loading() {
                            div { class: "flex flex-row gap-2 items-center justify-center",
                                p { class: "text-gray-600", "Loading Email Classifier..." }
                                // Improved spinner with a colored border and smoother animation
                                div {
                                    class: "w-6 h-6 border-4 border-blue-400 border-t-transparent rounded-full animate-spin mx-2",
                                    style: "border-top-color: transparent; border-right-color: #3b82f6; border-bottom-color: #3b82f6; border-left-color: #3b82f6;"
                                }
                            }
                        } else {
                            "Group by Category"
                        }
                    }
                }
            }

            if group_by_sender() {
                div { class: "bg-white p-8 rounded-2xl shadow-lg min-w-[350px] w-full m-10 text-center",
                    h1 { class: "text-2xl font-bold text-blue-700 mb-4", "Group by Sender" }
                    div { class: "flex flex-row gap-4 overflow-x-auto",
                        for (sender, messages) in sorted_msg_by_sender.iter().cloned() {
                            div { class: "bg-gray-100 p-4 rounded-lg flex flex-col gap-2",
                                style: "min-width: 600px;",
                                h2 { class: "text-lg font-bold text-blue-700 mb-2", "{sender} ({messages.len()})" }
                                button {
                                    class: "w-full py-3 bg-gray-300 text-gray-800 rounded-lg font-bold text-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-400",
                                    onclick: move |_| {
                                        let session_clone = session_signal().clone();
                                        let msg_uids = messages.iter().map(|m| m.uid).collect::<Vec<_>>();
                                        tokio::task::spawn_blocking(move || {
                                            let seq = msg_uids.iter().map(|uid| format!("{}", uid)).join(",");
                                            println!("Moving messages ({seq}) to Custodian/Trash");
                                            session_clone.lock().unwrap().mv(&seq, "Custodian/Trash").expect("Failed to move message to trash");
                                            println!("Moved messages ({seq}) to Custodian/Trash");
                                        });
                                    },
                                    "🗑️"
                                }
                                div { class: "max-h-96 overflow-y-auto flex flex-col gap-2",
                                    for message in messages.iter() {
                                        div { class: "bg-gray-200 p-2 rounded-lg w-full",
                                            p { class: "text-gray-600",  "[{message.uid}]" br {} b { "{message.subject}" } }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if group_by_category_active() {
                div { class: "bg-white p-8 rounded-2xl shadow-lg min-w-[350px] w-full m-10 text-center",
                    h1 { class: "text-2xl font-bold text-blue-700 mb-4", "Group by Category" }
                    h2 { class: "text-lg font-bold text-blue-700 mb-2", "Progress: {smart_progress()} / {local_msg_count}" }
                    div { class: "flex flex-row gap-4 overflow-x-auto",
                        for (category, messages) in sorted_msg_by_category.iter().cloned() {
                            div { class: "bg-gray-100 p-4 rounded-lg flex flex-col gap-2",
                                style: "min-width: 600px;",
                                h2 { class: "text-lg font-bold text-blue-700 mb-2", "{category} ({messages.len()})" }
                                button {
                                    class: r#"w-full py-3 rounded-lg font-bold text-lg transition-colors 
                                        duration-200 bg-gray-300 text-gray-800 
                                        enabled:hover:bg-gray-400
                                        disabled:bg-gray-200 disabled:text-gray-400 
                                        disabled:opacity-50 disabled:cursor-not-allowed"#,
                                    disabled: !smart_complete(),
                                    onclick: move |_| {
                                        let session_clone = session_signal().clone();
                                        let msg_uids = messages.iter().map(|m| m.uid).collect::<Vec<_>>();
                                        let ctgry = category.clone();
                                        tokio::task::spawn_blocking(move || {
                                            let seq = msg_uids.iter().map(|uid| format!("{}", uid)).join(",");
                                            session_clone.lock().unwrap().mv(seq, "Custodian/Trash").expect("Failed to move message to trash");
                                        });
                                    },
                                    "🗑️"
                                }
                                div { class: "max-h-96 overflow-y-auto flex flex-col gap-2",
                                    for message in messages.iter() {
                                        div { class: "bg-gray-200 p-2 rounded-lg min-w-[200px]",
                                            p { class: "text-gray-600",  "[{message.uid}]" br {} b {"{message.subject}"} }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
