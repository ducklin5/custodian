use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// User model for storing user information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub email: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Credentials model for storing saved login credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credentials {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub email: String,
    pub server: String,
    pub port: String,
    pub password: String, // In a real app, this should be encrypted
    pub name: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Email model for storing email data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Email {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub message_id: String,
    pub user_email: String,
    pub from_email: String,
    pub from_name: Option<String>,
    pub to_email: String,
    pub to_name: Option<String>,
    pub subject: String,
    pub body_text: String,
    pub body_html: Option<String>,
    pub date: DateTime<Utc>,
    pub is_read: bool,
    pub is_important: bool,
    pub is_spam: bool,
    pub mailbox: String,
    pub uid: u32,
    pub size: u32,
    pub flags: Vec<String>,
    pub attachments: Vec<Attachment>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Attachment model for email attachments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    pub filename: String,
    pub content_type: String,
    pub size: u32,
    pub content_id: Option<String>,
    pub disposition: String,
}

/// Category model for organizing emails
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Category {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    pub color: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// EmailCategory model for many-to-many relationship between emails and categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailCategory {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub email_id: String,
    pub category_id: String,
    pub created_at: DateTime<Utc>,
}

/// Email search filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailFilters {
    pub user_email: Option<String>,
    pub from_email: Option<String>,
    pub to_email: Option<String>,
    pub subject: Option<String>,
    pub mailbox: Option<String>,
    pub is_read: Option<bool>,
    pub is_important: Option<bool>,
    pub is_spam: Option<bool>,
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
    pub category_ids: Option<Vec<String>>,
}

/// Email sort options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmailSort {
    DateAsc,
    DateDesc,
    SubjectAsc,
    SubjectDesc,
    FromAsc,
    FromDesc,
    SizeAsc,
    SizeDesc,
}

/// Pagination options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pagination {
    pub page: u32,
    pub per_page: u32,
}

impl Default for Pagination {
    fn default() -> Self {
        Self {
            page: 1,
            per_page: 20,
        }
    }
}

/// Email search result with pagination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailSearchResult {
    pub emails: Vec<Email>,
    pub total: u64,
    pub page: u32,
    pub per_page: u32,
    pub total_pages: u32,
}

impl EmailSearchResult {
    pub fn new(emails: Vec<Email>, total: u64, pagination: Pagination) -> Self {
        let total_pages = (total as f64 / pagination.per_page as f64).ceil() as u32;
        Self {
            emails,
            total,
            page: pagination.page,
            per_page: pagination.per_page,
            total_pages,
        }
    }
}

/// User preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub user_email: String,
    pub theme: String,
    pub language: String,
    pub timezone: String,
    pub email_signature: Option<String>,
    pub auto_save_drafts: bool,
    pub auto_mark_read: bool,
    pub default_mailbox: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            id: None,
            user_email: String::new(),
            theme: "light".to_string(),
            language: "en".to_string(),
            timezone: "UTC".to_string(),
            email_signature: None,
            auto_save_drafts: true,
            auto_mark_read: false,
            default_mailbox: "INBOX".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
} 