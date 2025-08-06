use dioxus::prelude::*;
use crate::db::{CustodianDB, models::Credentials};
use crate::utils::AsyncPtrProp;

#[derive(PartialEq, Clone)]
pub enum ImapServer {
    Gmail,
    Yahoo,
    Outlook,
    Custom,
}

#[component]
pub fn LoginPage(
    db: AsyncPtrProp<CustodianDB>,
    on_login: EventHandler<(String, String, String, String, bool)>
) -> Element {
    let mut server = use_signal(|| ImapServer::Gmail);
    let mut custom_server = use_signal(|| String::new());
    let mut custom_port = use_signal(|| String::from("993"));
    let mut email = use_signal(|| String::new());
    let mut password = use_signal(|| String::new());
    let mut save_credentials = use_signal(|| false);
    let mut saved_credentials = use_signal(Vec::<Credentials>::new);
    let mut selected_profile = use_signal(|| String::new());

    // Load saved credentials on mount
    use_effect(move || {
        if let Ok(creds) = db.lock().unwrap().get_all_credentials() {
            saved_credentials.set(creds);
        }
    });

    // Function to select a saved credential
    let mut select_credential = move |cred: Credentials| {
        email.set(cred.email.clone());
        
        // Set server based on the saved server
        match cred.server.as_str() {
            "imap.gmail.com" => server.set(ImapServer::Gmail),
            "imap.mail.yahoo.com" => server.set(ImapServer::Yahoo),
            "imap-mail.outlook.com" => server.set(ImapServer::Outlook),
            _ => {
                server.set(ImapServer::Custom);
                custom_server.set(cred.server.clone());
                custom_port.set(cred.port.clone());
            }
        }
        
        password.set(cred.password.clone());
        selected_profile.set(cred.email.clone());
    };

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
                // Profiles selection
                div { class: "mb-4",
                    label { class: "block font-semibold mb-1", "Profiles:" }
                    select {
                        class: "w-full rounded-lg px-3 py-2 border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-400",
                        value: selected_profile,
                        onchange: move |e| {
                            let selected_email = e.value();
                            if !selected_email.is_empty() && selected_email != "select_profile" && selected_email != "no_profiles" {
                                if let Some(cred) = saved_credentials().iter().find(|c| c.email == selected_email) {
                                    select_credential(cred.clone());
                                }
                            }
                        },
                        if saved_credentials().is_empty() {
                            option { value: "no_profiles", selected: true, "No profiles available" }
                        } else {
                            option { value: "select_profile", selected: selected_profile().is_empty(), "Select a profile" }
                            for cred in saved_credentials() {
                                option { 
                                    value: "{cred.email}", 
                                    selected: selected_profile() == cred.email,
                                    "{cred.email}" 
                                }
                            }
                        }
                    }
                }
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
                div { class: "mb-4",
                    label { class: "block font-semibold mb-1", "Password:" }
                    input {
                        class: "w-full rounded-lg px-3 py-2 border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-400",
                        r#type: "password",
                        value: password,
                        oninput: move |e| password.set(e.value()),
                        placeholder: "Password"
                    }
                }
                // Save credentials checkbox
                div { class: "mb-6",
                    label { class: "flex items-center cursor-pointer",
                        input {
                            class: "mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded",
                            r#type: "checkbox",
                            checked: save_credentials,
                            onchange: move |e| save_credentials.set(e.checked()),
                        }
                        span { class: "text-sm text-gray-700", "Save credentials for future logins" }
                    }
                }
                // Login & Analyze button
                button {
                    class: "w-full py-3 bg-blue-700 text-white rounded-lg font-bold text-lg hover:bg-blue-800 transition-colors duration-200",
                    onclick: move |_| {
                        on_login.call((server_addr.clone(), port.clone(), email().clone(), password().clone(), save_credentials()));
                    },
                    "Login & Analyze"
                }
            }
        }
    }
} 