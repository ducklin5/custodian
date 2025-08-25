#![recursion_limit = "512"]

mod ai;
mod db;
mod pages;
mod utils;

use std::env;

use dioxus::prelude::*;
use dioxus_desktop::{tao::window::Icon, LogicalSize, WindowBuilder};

use pages::*;
use utils::AsyncPtrProp;

const ICON: Asset = asset!("/assets/favicon.ico");
const ICON_BYTES: &[u8] = include_bytes!("../assets/favicon.ico");

fn icon_from_bytes(bytes: &[u8]) -> Icon {
    let image = image::load_from_memory(bytes)
        .expect("Failed to load icon")
        .into_rgba8();
    let (width, height) = image.dimensions();
    Icon::from_rgba(image.into_raw(), width, height)
        .expect("Failed to load icon")
}

fn make_config() -> dioxus::desktop::Config {
    let config = dioxus::desktop::Config::default()
        .with_window(make_window())
        .with_icon(icon_from_bytes(ICON_BYTES));
     if cfg!(target_os = "windows") {
        let user_data_dir = env::var("LOCALAPPDATA").expect("env var LOCALAPPDATA not found");
        return config.with_data_directory(user_data_dir);
    }
    config
}

fn make_window() -> WindowBuilder {
    WindowBuilder::new().with_inner_size(LogicalSize::new(1024, 1024))
}
fn main() {
    LaunchBuilder::new().with_cfg(make_config()).launch(App);
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

#[component]
fn App() -> Element {
    rsx! {
        document::Title { "Custodian" }
        document::Stylesheet { key: "output", href: asset!("/assets/output.css") }
        document::Link { rel: "icon", r#type: "image/x-icon", href: ICON }
        InnerApp {}
    }
}

#[component]
fn InnerApp() -> Element {
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
            if db_loading() {
                div { "Loading database..." }
            } else {
                div { "Failed to initialize database: {db_error()}" }
            }
        };
    }

    return match page() {
        Page::Login => rsx! {
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
            OrganizerPage {
                session: session.clone(),
                email: connection.email.clone(),
                mailbox: mailbox.clone(),
                on_back: move |(_, _, _)| page.set(Page::Home { session: session.clone(), connection: connection.clone() }),
            }
        }
    }
}
