use dioxus::prelude::*;
use crate::utils::AsyncPtrProp;

#[derive(Clone, Props, PartialEq)]
pub struct HomePageProps {
    session: AsyncPtrProp<imap::Session<native_tls::TlsStream<std::net::TcpStream>>>,
    server: String,
    port: String,
    email: String,
    on_logout: EventHandler,
    on_mailbox_select: EventHandler<(AsyncPtrProp<imap::Session<native_tls::TlsStream<std::net::TcpStream>>>, String, String)>,
}

#[component]
pub fn HomePage(props: HomePageProps) -> Element {
    let HomePageProps {
        session,
        server,
        port,
        email,
        on_logout,
        on_mailbox_select
    } = props; 
    // fetch a list of mailboxes
    let mut mailboxes = use_signal(|| Vec::<String>::new());

    let session_ptr = session.clone();
    let session_ptr2 = session.clone();
    
    // Fetch mailboxes when component mounts
    use_effect(move || {
        if let Ok(mut session_guard) = session_ptr.lock() {
            if let Ok(mailbox_list) = (&mut *session_guard).list(None, Some("*")) {
                let mut mailbox_names: Vec<String> = mailbox_list
                    .iter()
                    .map(|mb| mb.name().to_string())
                    .collect::<Vec<String>>();
                // create custodian folders if they don't exist
                if !mailbox_names.contains(&"Custodian/Trash".to_string()) {
                    if !mailbox_names.contains(&"Custodian".to_string())  {
                        session_guard.create("Custodian").unwrap();
                    }
                    session_guard.create("Custodian/Trash").unwrap();
                    mailbox_names.push("Custodian/Trash".to_string());
                }
                mailboxes.set(mailbox_names);
            }
        }
    });

     rsx! {
        div { class: "min-h-screen flex flex-col items-center justify-center bg-slate-100 gap-6",
            div { class: "bg-white p-8 rounded-2xl shadow-lg min-w-[350px] w-full max-w-xl text-center",
                h1 { class: "text-2xl font-bold text-green-700 mb-4", "Your Mailboxes" }
                div { class: "mb-2",
                    strong { "Connection Details:" }
                    p { class: "mb-2", strong { "Server:" } " {server}" }
                    p { class: "mb-2", strong { "Port:" } " {port}" }
                    p { class: "mb-2", strong { "Email:" } " {email}" } 
                }
                div { class: "mb-2",
                    strong { "Select Mailbox:" }
                    div {
                        class: "flex flex-row gap-2 flex-wrap justify-center w-full",
                        for mailbox in mailboxes.iter() {
                            MailboxButton {
                                mailbox: mailbox.clone(),
                                on_mailbox_select: on_mailbox_select.clone(),
                                email: email.clone(),
                                session: session.clone(),
                            }
                        }
                    }
                }
                button {
                    class: "mt-6 w-full py-3 bg-gray-300 text-gray-800 rounded-lg font-bold text-lg hover:bg-gray-400 transition-colors duration-200",
                    onclick: move |_| {
                        on_logout.call(());
                        session_ptr2.lock().unwrap().logout().unwrap();
                    },
                    "Logout"
                }
            }
        }
    }
}

#[component]
fn MailboxButton(
    mailbox: String,
    email: String,
    session: AsyncPtrProp<imap::Session<native_tls::TlsStream<std::net::TcpStream>>>,
    on_mailbox_select: EventHandler<(AsyncPtrProp<imap::Session<native_tls::TlsStream<std::net::TcpStream>>>,String, String)>
) -> Element {
    rsx! {
        button { 
            class: "mb-1 p-2 bg-blue-700 text-white rounded-lg font-bold text-lg hover:bg-blue-800 transition-colors duration-200", 
            onclick: move |_| { on_mailbox_select.call((session.clone(), email.clone(), mailbox.clone())) }, 
            "{mailbox}" 
        }
    }
} 