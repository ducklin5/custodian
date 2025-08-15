pub fn email_classification_prompt(categories: Vec<String>, subject: &str, sender: &str) -> String {
    let categories_str = categories.join(", ");
    format!(
        r#"<prompt>
        <context>You are a precise email organizer. </context>
        <task>Assign 1 to 3 concise topic categories for this email based ONLY on its subject and sender.</task>
        <requirements>
            <requirement>Output ONLY a comma-separated list of categories. No other words, no explanations, no sentences.</requirement>
            <requirement>You must output atleast 1 category. If you cannot find a category, output "Uncategorized".</requirement>
            <requirement>Do not output more than 3 categories.</requirement>
            <requirement>Each category is exactly 1 word (letters only). No punctuation, no IDs, no email addresses.</requirement>
            <requirement>No duplicates. Prefer generalizable topics over vendor names when possible.</requirement>
            <requirement>You may use categories from the following list of existing categories: {categories_str}</requirement>
            <requirement>You may also create new categories that are not in the list of categories provided in the input.</requirement>
            <requirement>Do not use names of people as categories.</requirement>
            <requirement>DO NOT USE NAMES OF PEOPLE AS CATEGORIES.</requirement>
            <requirement>Do not ask questions. Do not request more information.</requirement>
        </requirements>
        <examples>
            <example>
                <input>
                    <subject>Your April invoice is ready</subject>
                    <sender>billing@acme.com</sender>
                </input>
                <output>Billing, Invoices</output>
            </example>
            <example>
                <input>
                    <subject>Flight booking confirmation - NYC to SF</subject>
                    <sender>notifications@delta.com</sender>
                </input>
                <output>Travel, Bookings, Itineraries</output>
            </example>
            <example>
                <input>
                    <subject>This week's product newsletter</subject>
                    <sender>news@stripe.com</sender>
                </input>
                <output>Newsletters, Marketing</output>
            </example>
        </examples>
    </prompt>
    <input>
        <subject>{subject}</subject>
        <sender>{sender}</sender>
    </input>
    <output>"#)
}

pub fn email_classification_with_body_prompt(categories: Vec<String>, subject: &str, sender: &str, body: &str) -> String {
    let categories_str = categories.join(", ");
    let body_trunc = if body.len() > 1000 { &body[..1000] } else { body };
    format!(
    r#"<prompt>
        <context>You are a precise email organizer. </context>
        <task>Assign 1 to 3 concise topic categories for this email based on its subject, sender, and body.</task>
        <requirements>
            <requirement>Output ONLY a comma-separated list of categories. No other words, no explanations, no sentences.</requirement>
            <requirement>You must output atleast 1 category. If you cannot find a category, output "Uncategorized".</requirement>
            <requirement>Do not output more than 3 categories.</requirement>
            <requirement>Each category is exactly 1 word (letters only). No punctuation, no IDs, no email addresses.</requirement>
            <requirement>No duplicates. Prefer generalizable topics over vendor names when possible.</requirement>
            <requirement>You may use categories from the following list of existing categories: {categories_str}</requirement>
            <requirement>You may also create new categories that are not in the list of categories provided in the input.</requirement>
            <requirement>Do not use names of people as categories.</requirement>
            <requirement>DO NOT USE NAMES OF PEOPLE AS CATEGORIES.</requirement>
            <requirement>Do not ask questions. Do not request more information.</requirement>
        </requirements>
        <examples>
            <example>
                <input>
                    <subject>Your April invoice is ready</subject>
                    <sender>billing@acme.com</sender>
                    <body>Your April invoice is ready. Please find the invoice attached.</body>
                </input>
                <output>Billing, Invoices</output>
            </example>
            <example>
                <input>
                    <subject>Flight booking confirmation - NYC to SF</subject>
                    <sender>notifications@delta.com</sender>
                    <body>Your flight booking confirmation - NYC to SF is ready. Please find the confirmation attached.</body>
                </input>
                <output>Travel, Bookings, Itineraries</output>
            </example>
            <example>
                <input>
                    <subject>This week's product newsletter</subject>
                    <sender>news@stripe.com</sender>
                    <body>This week's product newsletter is ready. Please find the newsletter attached.</body>
                </input>
                <output>Newsletters, Marketing</output>
            </example>
        </examples>
    </prompt>
    <input>
        <subject>{subject}</subject>
        <sender>{sender}</sender>
        <body>{body_trunc}</body>
    </input>
    <output>"#)
}