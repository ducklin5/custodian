pub fn format_complex_prompt(
    task_description: &str,
    specific_instructions: &str,
    context: Option<&str>,
) -> String {
    let mut formatted_prompt = String::new();
    
    formatted_prompt.push_str("<|user|>\n");
    
    // Add task description
    formatted_prompt.push_str(&format!("# Task: {}\n\n", task_description));
    
    // Add context if provided
    if let Some(ctx) = context {
        formatted_prompt.push_str(&format!("# Context:\n{}\n\n", ctx));
    }
    
    // Add specific instructions
    formatted_prompt.push_str(&format!("# Instructions:\n{}\n\n", specific_instructions));
    
    formatted_prompt.push_str("<|end|>\n<|assistant|>\n\n");
    
    formatted_prompt
}

pub fn email_classification_prompt(subject: &str, sender: &str) -> String {
    format!(
r#"Email Classification Task:

Classify the following email into 1-5 categories based on its subject and sender.
You maybe use 

Example 1:
Subject: Special Offer - 50% Off
Sender: sales@store.com
Categories: [Marketing, Promotions]

Example 2:
Subject: Meeting tomorrow at 2pm
Sender: boss@company.com
Categories: [Work, Notifications, Meetings]

Example 3:
Subject: Your invoice is ready
Sender: noreply@billing.com
Categories: [Finance, Invoices]

Example 4:
Subject: {}
Sender: {}
Categories: ["#,
        subject, sender
    )
}

