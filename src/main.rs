#![recursion_limit = "256"]



mod utils;
mod pages;
mod ai;
mod db;

use dioxus::prelude::*;
use dioxus_desktop::{WindowBuilder, LogicalSize};

use utils::AsyncPtrProp;
use pages::*;

fn make_config() -> dioxus::desktop::Config {
    dioxus::desktop::Config::default().with_window(make_window())
}

fn make_window() -> WindowBuilder {
    WindowBuilder::new()
        .with_inner_size(LogicalSize::new(1024, 1024))
}
fn main() {
    LaunchBuilder::new()
        .with_cfg(make_config())
        .launch(app);
}

#[derive(Clone, PartialEq)]
struct ConnectionInfo {
    server: String,
    port: String,
    email: String,
}

#[derive(PartialEq, Clone)]
enum Page {
    Login,
    AuthPage {
        connection: ConnectionInfo,
        password: String,
    },
    Home {
        session: AsyncPtrProp<imap::Session<native_tls::TlsStream<std::net::TcpStream>>>,
        connection: ConnectionInfo,
    },
    Organizer {
        session: AsyncPtrProp<imap::Session<native_tls::TlsStream<std::net::TcpStream>>>,
        connection: ConnectionInfo,
        mailbox: String,
    },
}

fn app() -> Element {
    let mut page = use_signal(|| Page::Login);
    let mut db_signal = use_signal(|| Option::<AsyncPtrProp<db::CustodianDB>>::None);
    let mut db_loading = use_signal(|| true);
    let mut db_error = use_signal(|| String::new());
    
    use_effect(move || {
        match db::init_database().map(|db| AsyncPtrProp::new(db)) {
            Ok(db_prop) => db_signal.set(Some(db_prop)),
            Err(e) => db_error.set(e.to_string()),
        }
        db_loading.set(false);
    });

    if db_signal().is_none() {
        return rsx! {
            document::Title { "Custodian" }
            document::Stylesheet { href: asset!("/assets/output.css") }
            if db_loading() {
                div { "Loading database..." }
            } else {
                div { "Failed to initialize database: {db_error()}" }
            }
        };
    }

    match page() {
        Page::Login => rsx! {
            document::Title { "Custodian" }
            document::Stylesheet { href: asset!("/assets/output.css") }
            LoginPage { 
                db: db_signal().unwrap(),
                on_login: {
                    let db = db_signal().unwrap();
                    move |(server, port, email, password, save_credentials): (String, String, String, String, bool)| {
                        // Save credentials if requested
                        if save_credentials {
                            let credentials = db::models::Credentials {
                                id: None,
                                email: email.clone(),
                                server: server.clone(),
                                port: port.clone(),
                                password: password.clone(),
                                name: None,
                                created_at: chrono::Utc::now(),
                                updated_at: chrono::Utc::now(),
                            };

                            if let Err(e) = db.lock().unwrap().create_credentials(credentials) {
                                eprintln!("Failed to save credentials: {}", e);
                            }
                        }
                        
                        page.set(Page::AuthPage {
                            password,
                            connection: ConnectionInfo {
                                server,
                                port,
                                email,
                            },
                        });
                    }
                }
            }
        },
        Page::AuthPage { connection, password } => rsx! {
            document::Title { "Authentication" }
            document::Stylesheet { href: asset!("/assets/output.css") }
            AuthPage {
                server: connection.server.clone(),
                port: connection.port.clone(),
                email: connection.email.clone(),
                password: password.clone(),
                on_logout: move || page.set(Page::Login),
                on_success: move |(session, _, _, _)| {
                    page.set(Page::Home {
                        session,
                        connection: connection.clone(),
                    });
                },
            }
        },
        Page::Home { session, connection } => rsx! {
            document::Title { "Home" }
            document::Stylesheet { href: asset!("/assets/output.css") }
            HomePage {
                session: session.clone(),
                server: connection.server.clone(),
                port: connection.port.clone(),
                email: connection.email.clone(),
                on_logout: move || page.set(Page::Login),
                on_mailbox_select: move |(session, _, mailbox)| {
                    page.set(Page::Organizer {
                        session,
                        connection: connection.clone(),
                        mailbox,
                    });
                },
            }
        },
        Page::Organizer { session, connection, mailbox } => rsx! {
            document::Title { "Organizer" }
            document::Stylesheet { href: asset!("/assets/output.css") }
            OrganizerPage {
                session: session.clone(),
                email: connection.email.clone(),
                mailbox: mailbox.clone(),
                on_back: move |(_, _, _)| page.set(Page::Home { session: session.clone(), connection: connection.clone() }),
            }
        },
    }
}


