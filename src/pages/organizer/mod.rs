mod message_card;

use dioxus::prelude::*;

use burn::backend::wgpu::WgpuDevice;

// use itertools::Itertools; // keep Itertools out to avoid deep generic types

use std::collections::HashMap;
use std::rc::Rc;

use crate::ai::models::llama::{BurnLlama, CppLlama, TextGenerator};
use crate::ai::prompts::email_classification_with_body_prompt;
use crate::utils::{
    AsyncPtrProp,
    organizer::{MessageInfo, fetch_messages, move_messages_to_trash},
    progress::{UiProgress, UiProgressSignal},
};
use message_card::MessageCard;

fn sanitize_label(raw: &str) -> String {
    // Remove anything after an XML-like opener
    let head = raw.split('<').next().unwrap_or("");
    // Explicitly strip common trailing tags if they slipped through
    let head = head
        .trim_end_matches("</output>")
        .trim_end_matches("/output>");
    // Clean up quotes, stray punctuation, and whitespace
    let cleaned = head
        .trim()
        .trim_matches(['"', '\'', '.', ';', ':', ' ', '\t', '\n', '\r'].as_ref());
    // Enforce single-token alphabetic labels and title-case them
    let token = cleaned.split_whitespace().next().unwrap_or("");
    let token = token.trim_matches(|c: char| !c.is_alphabetic());
    if token.is_empty() {
        return String::new();
    }
    let mut chars = token.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str().to_lowercase()),
        None => String::new(),
    }
}

fn is_valid_label(label: &str) -> bool {
    if label.len() < 3 || label.len() > 20 {
        return false;
    }
    if !label.chars().all(|c| c.is_alphabetic()) {
        return false;
    }
    true
}

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
    let session_signal = use_signal_sync(|| session.clone());
    let mailbox_signal = use_signal_sync(|| mailbox.clone());
    let mut group_by_sender = use_signal(|| false);
    let mut group_by_category_active = use_signal(|| false);
    let mut selected_uid = use_signal_sync(|| Option::<u32>::None);

    let mut message_elements = use_signal(|| HashMap::<u32, Rc<MountedData>>::new());

    let model_progress = use_signal_sync(|| UiProgress::new("Loading Cached", 4));

    // Performance tracking
    let mut processing_start_time = use_signal_sync(|| Option::<std::time::Instant>::None);
    let smart_progress = use_signal_sync(|| 0);
    let smart_complete = use_signal_sync(|| false);

    let mut messages_by_sender = use_signal_sync(|| HashMap::<String, Vec<MessageInfo>>::new());
    let messages_by_category = use_signal_sync(|| HashMap::<String, Vec<MessageInfo>>::new());

    // Generator loading state
    let mut generator_loading = use_signal(|| true);
    let mut generator =
        use_signal_sync(|| Option::<AsyncPtrProp<Box<dyn TextGenerator + Send>>>::None);

    // Fetch messages when component mounts
    use_effect(move || {
        spawn(async move {
            let result = tokio::task::spawn_blocking(move || {
                fetch_messages(
                    session_signal().clone(),
                    mailbox_signal(),
                    server_msg_count.clone(),
                    messages_signal.clone(),
                    false,
                )
            })
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
            let model_progress = UiProgressSignal(model_progress.clone());

            println!("Loading optimized model for email classification...");
            let join_result = tokio::task::spawn_blocking(
                move || -> anyhow::Result<AsyncPtrProp<Box<dyn TextGenerator + Send>>> {
                    let mut model: Box<dyn TextGenerator + Send> =
                        if std::env::var("USE_BURN").is_ok() {
                            let model_id = "HuggingFaceTB/SmolLM2-135M";
                            let revision = "main";
                            let model = BurnLlama::new(model_id, revision, device, model_progress)?;
                            Box::new(model)
                        } else {
                            let model_id = "unsloth/gemma-3-1b-it-GGUF:BF16";
                            let revision = "main";
                            let model = CppLlama::new(model_id, revision, model_progress)?;
                            Box::new(model)
                        };
                    model.add_terminator("\n");
                    model.add_terminator("<");
                    model.add_terminator("</output>");
                    Ok(AsyncPtrProp::new(model))
                },
            )
            .await;

            match join_result {
                Ok(Ok(model)) => {
                    generator.set(Some(model));
                    generator_loading.set(false);
                    println!("Optimized email classification model loaded successfully!");
                }
                Ok(Err(e)) => {
                    println!("Failed to load model: {:?}", e);
                    generator_loading.set(false);
                }
                Err(e) => {
                    println!("Failed to spawn blocking task: {:?}", e);
                    generator_loading.set(false);
                }
            }
        });
    });

    // Group by sender callback
    let group_by_sender_callback = use_callback(move |_| {
        let mut map: HashMap<String, Vec<MessageInfo>> = HashMap::new();
        for msg in messages_signal.read().iter() {
            map.entry(msg.sender.clone()).or_default().push(msg.clone());
        }
        messages_by_sender.set(map);
        group_by_sender.set(true);
    });

    // Group by category callback
    let group_by_category_callback = use_callback(move |_| {
        if !generator_loading() && generator.read().is_some() {
            group_by_category_active.set(true);
            let messages_clone = messages_signal.clone();
            let mut messages_by_category_clone = messages_by_category.clone();
            let generator_clone = generator.clone();
            let mut smart_progress_clone = smart_progress.clone();
            let mut smart_complete_clone = smart_complete.clone();

            processing_start_time.set(Some(std::time::Instant::now()));

            spawn(async move {
                let _result = tokio::task::spawn_blocking(move || {
                    let mut progress = 0;
                    let opt_model = generator_clone.read().clone();
                    if let Some(model_ptr) = opt_model {
                        let mut model_box = model_ptr.lock().unwrap();
                        for msg in messages_clone.read().iter() {
                            let mut groups = messages_by_category_clone.write();
                            let mut counts: Vec<(String, usize)> =
                                groups.iter().map(|(k, v)| (k.clone(), v.len())).collect();
                            counts.sort_by(|a, b| b.1.cmp(&a.1));
                            let top_10_categories = counts
                                .into_iter()
                                .take(10)
                                .map(|(k, _)| k)
                                .collect::<Vec<_>>();

                            let body_text = match crate::utils::organizer::fetch_message_body(
                                session_signal().clone(),
                                &mailbox_signal(),
                                msg.uid,
                            ) {
                                Ok(b) => b,
                                Err(_) => String::new(),
                            };
                            let prompt = email_classification_with_body_prompt(
                                top_10_categories,
                                &msg.subject,
                                &msg.sender,
                                &body_text,
                            );
                            let result = model_box
                                .as_mut()
                                .generate(prompt, None, None)
                                .unwrap_or("Err".into());
                            let mut labels = result
                                .split(|c| c == ',' || c == '\n' || c == ';' || c == '|')
                                .map(|s| sanitize_label(s))
                                .filter(|s| is_valid_label(s))
                                .take(3)
                                .collect::<Vec<_>>();
                            labels.sort();
                            labels.dedup();
                            if labels.is_empty() {
                                labels.push("Uncategorized".to_string());
                            }
                            println!("Labels: {:?}", labels);
                            for label in labels {
                                groups.entry(label).or_insert(Vec::new()).push(msg.clone());
                            }
                            progress += 1;
                            *smart_progress_clone.write() = progress;
                            std::thread::sleep(std::time::Duration::from_millis(10));
                        }
                    }
                    *smart_complete_clone.write() = true;
                })
                .await;
            });
        }
    });

    let local_msg_count = messages_signal.read().len();
    let sorted_msg_by_sender = {
        let msg_by_sender = messages_by_sender.read();
        let mut v: Vec<(String, Vec<MessageInfo>)> =
            msg_by_sender.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        v.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        v
    };

    let sorted_msg_by_category = {
        let msg_by_category = messages_by_category.read();
        let mut v: Vec<(String, Vec<MessageInfo>)> =
            msg_by_category.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        v.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        v
    };

    let current_model_progress = model_progress.read();

    // Compute performance stats outside RSX
    let (performance_info, progress_info) = if group_by_category_active() && !smart_complete() {
        let processed = smart_progress();
        let stats_str = format!("Processed: {}", processed);

        let rate_info = if let Some(start_time) = processing_start_time() {
            if let Some(elapsed) = std::time::Instant::now().checked_duration_since(start_time) {
                if elapsed.as_secs() > 0 {
                    let rate = smart_progress() as f64 / elapsed.as_secs() as f64;
                    let eta = if rate > 0.0 {
                        ((local_msg_count - smart_progress()) as f64 / rate).round() as u64
                    } else {
                        0
                    };
                    Some(format!("Rate: {:.1} emails/sec | ETA: {}s", rate, eta))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        (Some(stats_str), rate_info)
    } else {
        (None, None)
    };

    // Shared callbacks for message card actions
    let on_select_message = use_callback(move |uid: u32| {
        selected_uid.set(Some(uid));
        println!("Selected uid: {}", uid);
        if let Some(el) = message_elements().get(&uid) {
            let el = el.clone();
            spawn(async move {
                let _ = el.scroll_to(ScrollBehavior::Smooth).await;
            });
        }
    });

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
                                        th { class: "border-3 border-black px-4 py-2 text-left ", "Actions" }
                                    }
                                }
                                tbody {
                                    for message in messages_signal() {
                                        tr { 
                                            id: "row-{message.uid}", 
                                            class: "hover:bg-gray-50 cursor-pointer",
                                            class: if selected_uid() == Some(message.uid) { "bg-yellow-100 hover:bg-yellow-200" },
                                            onclick: move |_| selected_uid.set(Some(message.uid)),
                                            onmounted: move |ctx| {
                                                message_elements.write().insert(message.uid, ctx.data());
                                            },
                                            td { class: "border border-gray-300 px-4 py-2", "{message.uid}" }
                                            td { class: "border border-gray-300 px-4 py-2 text-left", "{message.subject}" }
                                            td { class: "border border-gray-300 px-4 py-2", "[{message.sender}] {message.sender_name}" }
                                            td { class: "border border-gray-300 px-4 py-2", "{message.date}" }
                                            td { class: "border border-gray-300 px-4 py-2",
                                                button {
                                                    class: "px-3 py-1 bg-gray-300 text-gray-800 rounded-lg font-bold hover:bg-gray-400 transition-colors duration-200",
                                                    onclick: move |_| {
                                                        let session_clone = session_signal().clone();
                                                        let uid = message.uid;
                                                        match move_messages_to_trash(session_clone, mailbox_signal(), vec![uid]) {
                                                            Ok(uids) => {
                                                                messages_signal.set(messages_signal().clone().into_iter().filter(|m| !uids.contains(&m.uid)).collect());
                                                            }
                                                            Err(e) => {
                                                                println!("Failed to move message: {}", e);
                                                            }
                                                        }
                                                    },
                                                    "🗑️"
                                                }
                                            }
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
                            group_by_sender_callback.call(());
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
                            div { class: "flex flex-col gap-2 items-center justify-center",
                                p { class: "text-gray-600", "Loading Email Classifier..." }
                                // Improved spinner with a colored border and smoother animation
                                div {
                                    class: "min-w-6 min-h-6 max-w-6 max-h-6 border-4 border-blue-400 border-t-transparent rounded-full animate-spin mx-2",
                                    style: "border-top-color: transparent; border-right-color: #3b82f6; border-bottom-color: #3b82f6; border-left-color: #3b82f6;"
                                }
                                div { class: "text-gray-600", "Stage: {current_model_progress.stage} ({current_model_progress.stages_done} / {current_model_progress.stages_total})" }
                                div { class: "text-gray-600", "Progress: {current_model_progress.progress} / {current_model_progress.total}" }
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
                                {
                                    let messages_for_delete = messages.clone();
                                    rsx! {
                                        button {
                                            class: "w-full py-3 bg-gray-300 text-gray-800 rounded-lg font-bold text-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-400",
                                            onclick: move |_| {
                                                let session_clone = session_signal().clone();
                                                let msg_uids = messages_for_delete.iter().map(|m| m.uid).collect::<Vec<_>>();
                                                if let Ok(uids) = move_messages_to_trash(session_clone, mailbox_signal(), msg_uids.clone()) {
                                                    messages_signal.set(messages_signal().clone().into_iter().filter(|m| !uids.contains(&m.uid)).collect());
                                                } else {
                                                    println!("Failed to move messages");
                                                }
                                            },
                                            "🗑️"
                                        }
                                    }
                                }
                                div { class: "max-h-96 overflow-y-auto flex flex-col gap-2",
                                    for message in messages {
                                        MessageCard {
                                            uid: message.uid,
                                            subject: message.subject.clone(),
                                            sender: message.sender.clone(),
                                            exists_in_inbox: messages_signal().iter().any(|m| m.uid == message.uid),
                                            is_selected: selected_uid() == Some(message.uid),
                                            enable_highlight: true,
                                            extra_class: "w-full".to_string(),
                                            on_select: on_select_message.clone(),
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

                    if !smart_complete() {
                        div { class: "mb-4 p-4 bg-blue-50 rounded-lg",
                            h2 { class: "text-lg font-bold text-blue-700 mb-2", "Processing: {smart_progress()} / {local_msg_count}" }

                            // Progress bar
                            div { class: "w-full bg-gray-200 rounded-full h-2.5 mb-2",
                                div {
                                    class: "bg-blue-600 h-2.5 rounded-full transition-all duration-300",
                                    style: "width: {(smart_progress() as f32 / local_msg_count as f32 * 100.0).min(100.0)}%"
                                }
                            }

                            // Performance stats
                            if let Some(stats) = &performance_info {
                                div { class: "text-sm text-gray-600 mt-2",
                                    p { "{stats}" }
                                    if let Some(rate_info) = &progress_info {
                                        p { "{rate_info}" }
                                    }
                                }
                            }
                        }
                    }

                    div { class: "flex flex-row gap-4 overflow-x-auto",
                        for (category, messages) in sorted_msg_by_category.iter().cloned() {
                            div { class: "bg-gray-100 p-4 rounded-lg flex flex-col gap-2",
                                style: "min-width: 600px;",
                                h2 { class: "text-lg font-bold text-blue-700 mb-2", "{category} ({messages.len()})" }
                                {
                                    let messages_for_delete = messages.clone();
                                    let category_for_delete = category.clone();
                                    rsx! {
                                        button {
                                            class: r#"w-full py-3 rounded-lg font-bold text-lg transition-colors
                                                duration-200 bg-gray-300 text-gray-800 
                                                enabled:hover:bg-gray-400
                                                disabled:bg-gray-200 disabled:text-gray-400 
                                                disabled:opacity-50 disabled:cursor-not-allowed"#,
                                            disabled: !smart_complete(),
                                            onclick: move |_| {
                                                let session_clone = session_signal().clone();
                                                let msg_uids = messages_for_delete.iter().map(|m| m.uid).collect::<Vec<_>>();
                                                let _ctgry = category_for_delete.clone();
                                                match move_messages_to_trash(session_clone, mailbox_signal(), msg_uids.clone()) {
                                                    Ok(uids) => {
                                                        messages_signal.set(messages_signal().clone().into_iter().filter(|m| !uids.contains(&m.uid)).collect());
                                                    }
                                                    Err(e) => {
                                                        println!("Failed to move message: {}", e);
                                                    }
                                                }
                                            },
                                            "🗑️"
                                        }
                                    }
                                }
                                div { class: "max-h-96 overflow-y-auto flex flex-col gap-2",
                                    for message in messages {
                                        MessageCard {
                                            uid: message.uid,
                                            subject: message.subject.clone(),
                                            sender: message.sender.clone(),
                                            exists_in_inbox: messages_signal().iter().any(|m| m.uid == message.uid),
                                            is_selected: selected_uid() == Some(message.uid),
                                            enable_highlight: true,
                                            extra_class: "min-w-[200px]".to_string(),
                                            on_select: on_select_message.clone(),
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
