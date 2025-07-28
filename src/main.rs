use dioxus::prelude::*;

mod utils;
mod pages;
mod ai;

use utils::AsyncPtrProp;
use pages::*;

fn main() {
    launch(app);
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

    match page() {
        Page::Login => rsx! {
            document::Title { "Custodian" }
            document::Stylesheet { href: asset!("/assets/output.css") }
            LoginPage { on_login: move |(server, port, email, password)| {
                page.set(Page::AuthPage {
                    password,
                    connection: ConnectionInfo {
                        server,
                        port,
                        email,
                    },
                });
            }}
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


