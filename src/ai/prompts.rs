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

pub fn email_classification_prompt(categories: Vec<String>, subject: &str, sender: &str) -> String {
    format!( r#"Classify the following email into 1-3 categories based on its subject and sender.

You can use a new category or use an existing category.
These are the existing categories: {:?}
New categories must be less that 3 words
The category list must be comma separated and must end with `<|end:categories|>` after the last category.

Example 1:
Subject: <|start:subject|> Meeting tomorrow at 2pm <|end:subject|>
Sender: <|start:sender|> boss@company.com <|end:sender|>
Categories: <|start:categories|> Work, Notifications, Meetings <|end:categories|>

Example 2:
Subject: <|start:subject|> Your invoice is ready <|end:subject|>
Sender: <|start:sender|> noreply@billing.com <|end:sender|>
Categories: <|start:categories|> Finance, Invoices <|end:categories|>

Task:
Subject: <|start:subject|> {} <|end:subject|>
Sender: <|start:sender|> {} <|end:sender|>
Categories: <|start:categories|>"#,
        categories, subject, sender
    )
}

