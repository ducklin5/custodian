use dioxus::prelude::*;
use dioxus::hooks::use_resource;
use native_tls::TlsConnector;
use crate::utils::AsyncPtrProp;

#[component]
pub fn AuthPage(
    server: String, 
    port: String, 
    email: String, 
    password: String, 
    on_logout: EventHandler,
    on_success: EventHandler<(AsyncPtrProp<imap::Session<native_tls::TlsStream<std::net::TcpStream>>>, String, String, String)>
) -> Element {
    // Clone these for display, so they're available after the closure
    let display_server = server.clone();
    let display_port = port.clone();
    let display_email = email.clone();
    let display_password = password.clone();

    let success_server = server.clone();
    let success_port = port.clone();
    let success_email = email.clone();

    let imap_session_resource = use_resource(move || {
        let server = server.clone();
        let port = port.clone();
        let email = email.clone();
        let password = password.clone();
        async move {
            let tls = match TlsConnector::builder().build() {
                Ok(tls) => tls,
                Err(e) => return Err(imap::error::Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e))),
            };

            let client = match imap::connect((server.as_str(), port.parse::<u16>().unwrap_or(993)), server.as_str(), &tls) {
                Ok(c) => c,
                Err(e) => return Err(e),
            };
            match client.login(&email, &password) {
                Ok(session) => {
                    Ok(AsyncPtrProp::new(session))
                },
                Err((e, _)) => Err(e),
            }
        }
    });

    use_effect(move || {
        if let Some(Ok(ref session_arc)) = *imap_session_resource.read() {
            on_success.call((session_arc.clone(), success_server.clone(), success_port.clone(), success_email.clone()));
        }
    });

    match *imap_session_resource.read() {
        None => rsx! {
            div { class: "min-h-screen flex flex-col items-center justify-center bg-slate-100 gap-6",
                div { class: "bg-white p-8 rounded-2xl shadow-lg min-w-[350px] w-full max-w-md text-center",
                    h1 { class: "text-2xl font-bold text-blue-700 mb-4", "Loading" }
                    p { class: "text-gray-600", "Attempting to log in to your mailbox..." }
                }
            }
        },
        Some(Ok(ref _session_arc)) => {
            rsx! {
                div { class: "min-h-screen flex flex-col items-center justify-center bg-slate-100 gap-6",
                    div { class: "bg-white p-8 rounded-2xl shadow-lg min-w-[350px] w-full max-w-md text-center",
                        h1 { class: "text-2xl font-bold text-green-700 mb-4", "Login Successful!" }
                        p { class: "text-gray-600", "Redirecting to your mailboxes..." }
                    }
                }
            }
        },
        Some(Err(ref e)) => rsx! {
            div { class: "min-h-screen flex flex-col items-center justify-center bg-slate-100 gap-6",
                div { class: "bg-white p-8 rounded-2xl shadow-lg min-w-[350px] w-full max-w-md text-center",
                    h1 { class: "text-2xl font-bold text-red-700 mb-4", "Failed to Login" }
                    p { class: "text-gray-600 mb-4", "Custodian failed to login to that server with the given credentials." }
                    p { class: "text-gray-600 mb-4", "Error: {e}" }
                    p { class: "text-gray-600 mb-4", "Email: {display_email}" }
                    p { class: "text-gray-600 mb-4", "Password: {display_password}" }
                    p { class: "text-gray-600 mb-4", "Server: {display_server}" }
                    p { class: "text-gray-600 mb-4", "Port: {display_port}" }
                    button {
                        class: "w-full py-3 bg-blue-700 text-white rounded-lg font-bold text-lg hover:bg-blue-800 transition-colors duration-200",
                        onclick: move |_| on_logout.call(()),
                        "Return to Login"
                    }
                }
            }
        },
    }
} 