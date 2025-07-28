use dioxus::prelude::*;

#[derive(PartialEq, Clone)]
pub enum ImapServer {
    Gmail,
    Yahoo,
    Outlook,
    Custom,
}

#[component]
pub fn LoginPage(on_login: EventHandler<(String, String, String, String)>) -> Element {
    let mut server = use_signal(|| ImapServer::Gmail);
    let mut custom_server = use_signal(|| String::new());
    let mut custom_port = use_signal(|| String::from("993"));
    let mut email = use_signal(|| String::new());
    let mut password = use_signal(|| String::new());

    let (server_addr, port) = match server() {
        ImapServer::Gmail => (String::from("imap.gmail.com"), String::from("993")),
        ImapServer::Yahoo => (String::from("imap.mail.yahoo.com"), String::from("993")),
        ImapServer::Outlook => (String::from("imap-mail.outlook.com"), String::from("993")),
        ImapServer::Custom => (custom_server().clone(), custom_port().clone()),
    };

    let custom_fields = if server() == ImapServer::Custom {
        Some(rsx! {
            div { class: "mb-4",
                label { class: "block font-semibold mb-1", "Server Address:" }
                input {
                    class: "w-full rounded-lg px-3 py-2 border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-400",
                    r#type: "text",
                    value: custom_server,
                    oninput: move |e| custom_server.set(e.value()),
                    placeholder: "imap.example.com"
                }
            }
            div { class: "mb-4",
                label { class: "block font-semibold mb-1", "Port:" }
                input {
                    class: "w-full rounded-lg px-3 py-2 border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-400",
                    r#type: "text",
                    value: custom_port,
                    oninput: move |e| custom_port.set(e.value()),
                    placeholder: "993"
                }
            }
        })
    } else {
        None
    };

    let gmail_note = if server() == ImapServer::Gmail {
        Some(rsx! {
            p {
                class: "mt-2 text-sm text-yellow-700 bg-yellow-100 rounded px-2 py-1 border border-yellow-300",
                "You will need to create an app password called ",
                strong { "Custodian" },
                " for your Google account here: "
                a {
                    href: "https://myaccount.google.com/apppasswords",
                    class: "underline text-blue-700 hover:text-blue-900",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "https://myaccount.google.com/apppasswords"
                }
            }
        })
    } else {
        None
    };
    let yahoo_note = if server() == ImapServer::Yahoo {
        Some(rsx! {
            p {
                class: "mt-2 text-sm text-yellow-700 bg-yellow-100 rounded px-2 py-1 border border-yellow-300",
                "You will need to create an app password called ",
                strong { "Custodian" },
                " for your Yahoo account here: "
                a {
                    href: "https://login.yahoo.com/myaccount/security/app-password/",
                    class: "underline text-blue-700 hover:text-blue-900",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "https://login.yahoo.com/myaccount/security/app-password/"
                }
                "Please note that Yahoo's IMAP server is limited to 10000 messages. Emails may be missing for large mailboxes."
            }
        })
    } else {
        None
    };
    rsx! {
        div { class: "min-h-screen flex items-center justify-center bg-slate-100",
            div { class: "bg-white p-8 rounded-2xl shadow-lg min-w-[350px] w-full max-w-md",
                h1 { class: "text-2xl font-bold text-center mb-6 text-blue-700", "Custodian" }
                // Server selection
                div { class: "mb-4",
                    label { class: "block font-semibold mb-1", "IMAP Server:" }
                    select {
                        class: "w-full rounded-lg px-3 py-2 border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-400",
                        value: match server() {
                            ImapServer::Gmail => "gmail",
                            ImapServer::Yahoo => "yahoo",
                            ImapServer::Outlook => "outlook",
                            ImapServer::Custom => "custom",
                        },
                        onchange: move |e| {
                            match e.value().as_str() {
                                "gmail" => server.set(ImapServer::Gmail),
                                "yahoo" => server.set(ImapServer::Yahoo),
                                "outlook" => server.set(ImapServer::Outlook),
                                "custom" => server.set(ImapServer::Custom),
                                _ => {},
                            }
                        },
                        option { value: "gmail", "Google (Gmail)" }
                        option { value: "yahoo", "Yahoo" }
                        option { value: "outlook", "Outlook" }
                        option { value: "custom", "Custom" }
                    }
                    {gmail_note}
                    {yahoo_note}
                }
                {custom_fields}
                // Email input
                div { class: "mb-4",
                    label { class: "block font-semibold mb-1", "Email:" }
                    input {
                        class: "w-full rounded-lg px-3 py-2 border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-400",
                        r#type: "email",
                        value: email,
                        oninput: move |e| email.set(e.value()),
                        placeholder: "you@email.com"
                    }
                }
                // Password input
                div { class: "mb-6",
                    label { class: "block font-semibold mb-1", "Password:" }
                    input {
                        class: "w-full rounded-lg px-3 py-2 border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-400",
                        r#type: "password",
                        value: password,
                        oninput: move |e| password.set(e.value()),
                        placeholder: "Password"
                    }
                }
                // Login & Analyze button
                button {
                    class: "w-full py-3 bg-blue-700 text-white rounded-lg font-bold text-lg hover:bg-blue-800 transition-colors duration-200",
                    onclick: move |_| {
                        on_login.call((server_addr.clone(), port.clone(), email().clone(), password().clone()));
                    },
                    "Login & Analyze"
                }
            }
        }
    }
} 