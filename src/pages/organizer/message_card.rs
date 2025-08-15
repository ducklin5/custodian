use dioxus::prelude::*;

#[derive(Clone, Props, PartialEq)]
pub struct MessageCardProps {
    uid: u32,
    subject: String,
    sender: String,
    exists_in_inbox: bool,
    is_selected: bool,
    enable_highlight: bool,
    extra_class: String,
    on_select: EventHandler<u32>,
}

#[component]
pub fn MessageCard(props: MessageCardProps) -> Element {
    let MessageCardProps {
        uid,
        subject,
        sender,
        exists_in_inbox,
        is_selected,
        enable_highlight,
        extra_class,
        on_select,
    } = props;

    let mut classes = String::from("bg-gray-200 p-2 rounded-lg cursor-pointer");
    if !extra_class.is_empty() {
        classes.push(' ');
        classes.push_str(&extra_class);
    }
    if enable_highlight && is_selected {
        classes.push_str(" bg-yellow-100");
    }

    rsx! {
        div { class: "{classes}",
            onclick: move |_| async move {
                on_select.call(uid);
            },
            div { class: "text-gray-600 flex flex-col gap-1",
                div { class: "flex flex-row gap-2 items-center justify-center",
                    "[{uid}]"
                    if exists_in_inbox {
                        div { class: "w-5 h-5 bg-green-500 rounded-full", style: "margin-right: 4px;" }
                    } else {
                        div { class: "w-5 h-5 bg-red-500 rounded-full", style: "margin-right: 4px;" }
                    }
                }
                b { "{subject}" }
                span { "{sender}" }
            }
        }
    }
}

