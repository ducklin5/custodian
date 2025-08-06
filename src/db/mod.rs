use std::path::Path;

use anyhow::{Context, Result};
use chrono::{Utc};
use polodb_core::{
    bson::{to_document, doc},
    Collection, CollectionT, Database,
};
use directories::ProjectDirs;

pub mod models;

use models::*;

/// PoloDB-based database for Custodian
pub struct CustodianDB {
    db: Database,
}

impl CustodianDB {
    /// Create a new database instance
    pub fn new(db_path: &str) -> Result<Self> {
        let db_path = Path::new(db_path);
        if !db_path.exists() {
            std::fs::create_dir_all(db_path).context("Failed to create database directory")?;
        }
        
        let db = Database::open_path(db_path).context("Failed to open database")?;
        Ok(Self { db })
    }

    /// Initialize the database with default data
    pub fn init(&self) -> Result<()> {
        // Add default categories if they don't exist
        let categories_collection = self.db.collection("categories");

        // Check if categories already exist
        let existing_categories = categories_collection.count_documents()?;

        if existing_categories == 0 {
            let default_categories = vec![
                ("Inbox", "#3B82F6"),
                ("Sent", "#10B981"),
                ("Drafts", "#F59E0B"),
                ("Trash", "#EF4444"),
                ("Important", "#8B5CF6"),
                ("Work", "#06B6D4"),
                ("Personal", "#84CC16"),
                ("Spam", "#F97316"),
            ];

            for (name, color) in default_categories {
                let category = Category {
                    id: None,
                    name: name.to_string(),
                    color: color.to_string(),
                    description: None,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                };

                categories_collection.insert_one(category)?;
            }
        }

        Ok(())
    }

    // User operations
    pub fn find_user_by_email(&self, email: &str) -> Result<Option<User>> {
        let collection: Collection<User> = self.db.collection("users");
        let filter = doc! { "email": email };
        collection
            .find_one(filter)
            .context("Failed to find user by email")
    }

    pub fn create_user(&mut self, mut user: User) -> Result<User> {
        let collection: Collection<User> = self.db.collection("users");
        user.id = Some(uuid::Uuid::new_v4().to_string());
        user.created_at = Utc::now();
        user.updated_at = Utc::now();

        let result = collection.insert_one(&user)?;

        if let Some(id) = result.inserted_id.as_str() {
            user.id = Some(id.to_string());
        }

        Ok(user)
    }

    pub fn update_user(&mut self, mut user: User) -> Result<User> {
        let collection: Collection<User> = self.db.collection("users");
        user.updated_at = Utc::now();

        if let Some(id) = &user.id {
            let filter = doc! { "_id": id };
            let update = doc! { "$set": to_document(&user)? };
            collection.update_one(filter, update)?;
        }

        Ok(user)
    }

    pub fn delete_user(&mut self, user_id: &str) -> Result<bool> {
        let collection: Collection<User> = self.db.collection("users");
        let filter = doc! { "_id": user_id };
        let result = collection.delete_one(filter)?;
        Ok(result.deleted_count > 0)
    }

    // Credentials operations
    pub fn find_credentials_by_email(&self, email: &str) -> Result<Option<Credentials>> {
        let collection: Collection<Credentials> = self.db.collection("credentials");
        let filter = doc! { "email": email };
        collection
            .find_one(filter)
            .context("Failed to find credentials by email")
    }

    pub fn find_credentials_by_email_and_server(
        &self,
        email: &str,
        server: &str,
    ) -> Result<Option<Credentials>> {
        let collection: Collection<Credentials> = self.db.collection("credentials");
        let filter = doc! { "email": email, "server": server };
        collection
            .find_one(filter)
            .context("Failed to find credentials by email and server")
    }

    pub fn create_credentials(&mut self, mut credentials: Credentials) -> Result<Credentials> {
        let collection: Collection<Credentials> = self.db.collection("credentials");
        credentials.id = Some(uuid::Uuid::new_v4().to_string());
        credentials.created_at = Utc::now();
        credentials.updated_at = Utc::now();

        let result = collection.insert_one(&credentials)?;

        if let Some(id) = result.inserted_id.as_str() {
            credentials.id = Some(id.to_string());
        }

        Ok(credentials)
    }

    pub fn update_credentials(&mut self, mut credentials: Credentials) -> Result<Credentials> {
        let collection: Collection<Credentials> = self.db.collection("credentials");
        credentials.updated_at = Utc::now();

        if let Some(id) = &credentials.id {
            let filter = doc! { "_id": id };
            let update = doc! { "$set": to_document(&credentials)? };
            collection.update_one(filter, update)?;
        }

        Ok(credentials)
    }

    pub fn delete_credentials(&mut self, credentials_id: &str) -> Result<bool> {
        let collection: Collection<Credentials> = self.db.collection("credentials");
        let filter = doc! { "_id": credentials_id };
        let result = collection.delete_one(filter)?;
        Ok(result.deleted_count > 0)
    }

    pub fn get_all_credentials(&self) -> Result<Vec<Credentials>> {
        let collection: Collection<Credentials> = self.db.collection("credentials");
        let cursor = collection.find(doc! {}).run()?;
        Ok(cursor.into_iter().filter_map(|doc| doc.ok()).collect())
    }

    // Email operations
    pub fn find_email_by_message_id(&self, message_id: &str) -> Result<Option<Email>> {
        let collection: Collection<Email> = self.db.collection("emails");
        let filter = doc! { "message_id": message_id };
        collection
            .find_one(filter)
            .context("Failed to find email by message id")
    }

    pub fn create_email(&mut self, mut email: Email) -> Result<Email> {
        let collection: Collection<Email> = self.db.collection("emails");
        email.id = Some(uuid::Uuid::new_v4().to_string());
        email.created_at = Utc::now();
        email.updated_at = Utc::now();

        let result = collection.insert_one(&email)?;

        if let Some(id) = result.inserted_id.as_str() {
            email.id = Some(id.to_string());
        }

        Ok(email)
    }

    pub fn update_email(&mut self, mut email: Email) -> Result<Email> {
        let collection: Collection<Email> = self.db.collection("emails");
        email.updated_at = Utc::now();

        if let Some(id) = &email.id {
            let filter = doc! { "_id": id };
            let update = doc! { "$set": to_document(&email)? };
            collection.update_one(filter, update)?;
        }

        Ok(email)
    }

    pub fn delete_email(&mut self, email_id: &str) -> Result<bool> {
        let collection: Collection<Email> = self.db.collection("emails");
        let filter = doc! { "_id": email_id };
        let result = collection.delete_one(filter)?;
        Ok(result.deleted_count > 0)
    }

    pub fn find_emails_by_user(&self, user_email: &str) -> Result<Vec<Email>> {
        let collection: Collection<Email> = self.db.collection("emails");
        let filter = doc! { "user_email": user_email };
        let cursor = collection.find(filter).run()?;

        Ok(cursor.into_iter().filter_map(|doc| doc.ok()).collect())
    }

    // Category operations
    pub fn find_category_by_name(&self, name: &str) -> Result<Option<Category>> {
        let collection: Collection<Category> = self.db.collection("categories");
        let filter = doc! { "name": name };
        collection
            .find_one(filter)
            .context("Failed to find category by name")
    }

    pub fn create_category(&mut self, mut category: Category) -> Result<Category> {
        let collection: Collection<Category> = self.db.collection("categories");
        category.id = Some(uuid::Uuid::new_v4().to_string());
        category.created_at = Utc::now();
        category.updated_at = Utc::now();

        let result = collection.insert_one(&category)?;

        if let Some(id) = result.inserted_id.as_str() {
            category.id = Some(id.to_string());
        }

        Ok(category)
    }

    pub fn update_category(&mut self, mut category: Category) -> Result<Category> {
        let collection: Collection<Category> = self.db.collection("categories");
        category.updated_at = Utc::now();

        if let Some(id) = &category.id {
            let filter = doc! { "_id": id };
            let update = doc! { "$set": to_document(&category)? };
            collection.update_one(filter, update)?;
        }

        Ok(category)
    }

    pub fn delete_category(&mut self, category_id: &str) -> Result<bool> {
        let collection: Collection<Category> = self.db.collection("categories");
        let filter = doc! { "_id": category_id };
        let result = collection.delete_one(filter)?;
        Ok(result.deleted_count > 0)
    }

    pub fn get_all_categories(&self) -> Result<Vec<Category>> {
        let collection: Collection<Category> = self.db.collection("categories");
        let cursor = collection.find(doc! {}).run()?;
        Ok(cursor.into_iter().filter_map(|doc| doc.ok()).collect())
    }

    // Email-Category relationship operations
    pub fn add_email_to_category(&mut self, email_id: &str, category_id: &str) -> Result<()> {
        let collection: Collection<EmailCategory> = self.db.collection("email_categories");
        let email_category = EmailCategory {
            id: Some(uuid::Uuid::new_v4().to_string()),
            email_id: email_id.to_string(),
            category_id: category_id.to_string(),
            created_at: Utc::now(),
        };

        collection.insert_one(&email_category)?;
        Ok(())
    }

    pub fn remove_email_from_category(&mut self, email_id: &str, category_id: &str) -> Result<()> {
        let collection: Collection<EmailCategory> = self.db.collection("email_categories");
        let filter = doc! { "email_id": email_id, "category_id": category_id };
        collection.delete_one(filter)?;
        Ok(())
    }

    pub fn get_emails_in_category(&self, category_id: &str) -> Result<Vec<Email>> {
        // First get all email IDs in this category
        let email_categories_collection: Collection<EmailCategory> =
            self.db.collection("email_categories");
        let filter = doc! { "category_id": category_id };
        let cursor = email_categories_collection.find(filter).run()?;

        let email_ids: Vec<String> = cursor
            .into_iter()
            .filter_map(|doc| doc.ok())
            .map(|doc| doc.email_id)
            .collect();

        if email_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Then get all emails with those IDs
        let emails_collection: Collection<Email> = self.db.collection("emails");
        let filter = doc! { "_id": { "$in": email_ids } };
        let cursor = emails_collection.find(filter).run()?;

        Ok(cursor.into_iter().filter_map(|doc| doc.ok()).collect())
    }

    pub fn get_categories_for_email(&self, email_id: &str) -> Result<Vec<Category>> {
        // First get all category IDs for this email
        let email_categories_collection: Collection<EmailCategory> =
            self.db.collection("email_categories");
        let filter = doc! { "email_id": email_id };
        let cursor = email_categories_collection
            .find(filter)
            .run()
            .context("Failed to get categories for email")?;

        let category_ids: Vec<String> = cursor
            .into_iter()
            .filter_map(|doc| doc.ok())
            .map(|doc| doc.category_id)
            .collect();

        if category_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Then get all categories with those IDs
        let categories_collection: Collection<Category> = self.db.collection("categories");
        let filter = doc! { "_id": { "$in": category_ids } };
        let cursor = categories_collection
            .find(filter)
            .run()
            .context("Failed to get categories for email")?;

        Ok(cursor.into_iter().filter_map(|doc| doc.ok()).collect())
    }
}

/// Initialize the database
pub fn init_database() -> Result<CustodianDB> {
    // use the app data directory
    let proj_dir = ProjectDirs::from("dev", "Alkove", "Custodian").context("Failed to get project directory")?;
    let db_path = proj_dir.data_local_dir().join("v1");
    let db_path = db_path.to_str().context("Failed to convert database path to string")?;
    let db = CustodianDB::new(db_path).context("Failed to create database")?;
    db.init()?;
    Ok(db)
}