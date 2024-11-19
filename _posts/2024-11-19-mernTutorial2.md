---
title: MERN Tutorial 2 - Backend
date: 2024-11-18 18:00:00 +0000
categories: [MERN Tutorial]
tags: [MERN, CRUD, Website Tutorial, Backend, MongoDB, Express, Node, ]
math: true
---

# Introduction

Welcome to the second edition of the MERN Tutorial series. In this blog, we will continue from our project initialisation in the MERN Tutorial 1 - Initialisation blog amd start implementing MongoDB and express into our application. Let us begin. 

# Step 1: Set Up MongoDB

## Step 1.1: Sign Up for MongoDB Atlas

Before we start implementing our database we must first create an account with MongoDB Atlas and sign in.

## Step 1.2: Create a Cluster

Once signed into MongoDB, you will see the overview like this:

![MongoDB Overview](assets/mern-tutorial/MongoDB-Overview.jpg)

Now, we must create a cluster. Click create a cluster and you will have these fields to choose from. Its not too important what you choose:

- Choose a Cluster Name
- Choose a Provider
- Choose Configuration Option (You can only have one free cluster)

## Step 1.3: Connect to application

Now we have created our cluster we need to connect to it. You should see a screen like this:

![Connect to Application](assets/mern-tutorial/Connect-Application.jpg)

Click Drivers under Connect to Application. You should now see a screen like this:

![Connection String](assets/mern-tutorial/Connection-String.jpg)

Since we are using mongoose we do not need to install MongoDB. However, we will need the connection string. Copy the connection string and in your .env file type MONGO_URI= and then your connection string like this:

```
MONGO_URI=mongodb+srv://robertwoodhams:<db_password>@cluster0.c47op.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
```

While were in the .env file we can specify the port we want to run our server on during development. We can write it like this:

```
PORT=5000
```

This connection string will CRUD data into a database called test automatically. However, if we want to choose another database to connect to we edit the connection string like so:

```
MONGO_URI=mongodb+srv://robertwoodhams:<db_password>@cluster0.c47op.mongodb.net/tasksdb?retryWrites=true&w=majority&appName=Cluster0
PORT=5000
```

Now our connection string will CRUD data into a database called tasksdb.

Now that we have successfully connected MongoDB to our application, now we can use our connection string to perform CRUD (Create, Read, Update, Delete) actions.

# Step 2: Backend Setup

## Step 2.1: Import Dependencies

To start our CRUD application we first need to import the dependencies we will be using. Open the index.js file and type:

```javascript
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();
```

### Code Explanation

- <strong> const express = require('express'); </strong>

    This line imports the express library, which is used to create and manage a web server in Node.js. It allows us to define routes (for example,  /, /api) and handle HTTP requests (GET, POST, PUT, DELETE). This serves as the foundation for your backend API.

- <strong> const mongoose = require('mongoose'); </strong>

    This line imports the mongoose library, which is an Object Data Modelling (ODM) tool for MongoDB. It simplifies interactions with the MongoDB database by providing schemas, models, and query-building tools. It enables structured data handling and database queries in your application.

- <strong> const cors = require('cors'); </strong>

    his line imports the cors middleware, which stands for Cross-Origin Resource Sharing. It allows our backend server to handle requests from other origins (for example, our frontend hosted on a different domain). It prevents "CORS errors" when the frontend (React) makes API calls to the backend.

- <strong> require('dotenv').config(); </strong>

    This line loads environment variables from our .env file into process.env. It enables secure storage of sensitive information, such as database connection strings (MONGO_URI), API keys, and ports.

## Step 2.2: Define Middleware

We now need to set up our middleware and enable Cross-Origin Resource Sharing (CORS) so that other domains (like our frontend) can access the backend.

```javascript
const app = express();
app.use(cors());
app.use(express.json());
```

### Code Explanation 

- <strong> const app = express(); </strong>

    This line creates an instance of an Express application. The app instance is used to define routes, middleware, and other configurations for the server. Basically it acts as the main object to manage the backend server.

- <strong> app.use(cors()); </strong>

    This line adds the cors middleware to the application. This allows our backend to handle requests from different origins (for example, a React frontend running on localhost:3000 making requests to a Node.js backend on localhost:5000).

    Without this line, you would encounter CORS errors in the browser when making API calls from the frontend to the backend.

- <strong> app.use(express.json()); </strong>

    This line adds the express.json() middleware to the application. It automatically parses incoming requests with a JSON payload and makes the parsed data available in req.body.

    It is essential for handling RESTful API requests where data is sent in JSON format, such as in POST or PUT requests.

## Step 2.3: Conect to MongoDB Atlas

We now need to connect our application to a MongoDB database using Mongoose, a library for interacting with MongoDB in Node.js.

```javascript
mongoose.connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
.then(() => console.log('Connected to MongoDB Atlas'))
.catch((err) => console.error('Error connecting to MongoDB:', err));
```

### Code Explanation

- <strong> mongoose.connect(process.env.MONGO_URI, {...}) </strong>

    This line initiates the connection to MongoDB using the URI stored in process.env.MONGO_URI.

- <strong> process.env.MONGO_URI </strong>

    This is our connection string for the MongoDB instance that we saved in the .env file.

- <strong> useNewUrlParser: true </strong>

    This ensures Mongoose uses the MongoDB driver’s newer connection string parser.

- <strong> useUnifiedTopology: true </strong>

    This enables the new server discovery and monitoring engine for improved stability.

- <strong> .then(() => console.log('Connected to MongoDB Atlas')) </strong>

    This line logs a message ('Connected to MongoDB Atlas') to indicate the connection was established.

- <strong> .catch((err) => console.error('Error connecting to MongoDB:', err)); </strong>

    This line Logs an error message and the error details (err) if there’s an error during the connection attempt.

## Step 2.3: Create Schema and Model

We now need to create our schema and model. 

### What is a Schema?

A schema is a blueprint or structure that defines how the data in a MongoDB collection should be organised. It specifies:

- Fields: The names of the fields in the document.
- Data Types: The type of data each field should store (for exampple, string, number, date, etc)
- Validation Rules: Optional constraints for each field

Since were only creating a basic CRUD application the only field for our schema will be the title. 

### What is a Model?

A model is a wrapper around the schema that provides an interface to interact with the database. It allows us to:

- Create new documents
- Read documents from the database
- Update existing documents
- Delete documents

```javascript
const taskSchema = new mongoose.Schema({
    title: { type: String, required: true }, 
});

const Task = mongoose.model('Task', taskSchema);
```

Here we created a schema named taskSchema and a model named Task. However, this is simply a variable name and you can name it whatever you like, as long as it adheres to JavaScript naming conventions.

### Code Explanation

- <strong> const taskSchema = new mongoose.Schema({...}); </strong>

This line defines the structure (schema) of a "Task" document in the MongoDB database.

- <strong> title: { type: String, required: true },  </strong>

    This line specifies the schema field title. 

    - type: String means that the field must be a string. 

    - required: true ensures that this field cannot be empty when creating a new task.

- <strong> const Task = mongoose.model('Task', taskSchema);  </strong>

    This creates a model from the taskSchema. The Task model provides an interface to interact with the "tasks" collection in MongoDB. It allows us to perform CRUD operations (Create, Read, Update, Delete) on "Task" documents.

    - 'Task': The name of the model. Mongoose automatically pluralises this name to create the collection name (tasks) in the database.

    - taskSchema: The schema used to define the structure of the documents in the "tasks" collection.

# Step 3: CRUD Implementation

We now have all the necessary steps to create, read, update and delete data from our MongoDB database. Let us now implement them to finalise our backend.

## Step 3.1: Implement the Create Operation

The create operation will be used to add new data to the database. Here is how to implement it:

```javascript
app.post('/tasks', async (req, res) => {
    const { title } = req.body;

    try {
        const newTask = new Task({ title });
        await newTask.save();
        res.status(201).json(newTask);
    } catch (err) {
        res.status(500).json({ error: 'Failed to create task' });
    }
});
```

### Code Explanation

- <strong> app.post('/tasks', async (req, res) => {...}); </strong>

    This line sets up an express POST route for the endpoint /tasks. The client can send a POST request to /tasks to create a new task.

    Async/Await: Makes the function asynchronous, allowing for the use of await with database operations.

    req (Request): Contains information about the HTTP request, such as headers, body, and parameters.

    res (Response): Used to send a response back to the client.

- <strong> const { title } = req.body; </strong>

    This line destructures the title field from the request body. It extracts the title so it can be used to create a new task.

- <strong> try {..} </strong>

    This wraps the task creation logic in a try block to handle potential errors. It ensures that any errors during database interaction or task creation are caught.

- <strong> const newTask = new Task({ title }); </strong>

    This creates a new instance of the Task model with the title field set to the value provided by the client.

- <strong> await newTask.save(); </strong>

    This line uses the .save() method in Mongoose to save the new task to the MongoDB database. 

    await: Waits for the database operation to complete before proceeding.

- <strong> res.status(201).json(newTask); </strong>

    This sends a JSON response back to the client with the newly created task and sets the HTTP status code to 201 (Created).

- <strong> catch (err) {...} </strong>

    This catches any errors that occur during the task creation process (for example, database connection issues or invalid data).

- <strong> res.status(500).json({ error: 'Failed to create task' }); </strong>

    This sends an error response to the client with an HTTP status code of 500 (Internal Server Error). It notifies the client that the task creation failed and provides an error message.


## Step 3.2: Implement the Read Operation

The read operation will be used to retrieve data from the database. Here is how to implement it:

```javascript
app.get('/tasks', async (req, res) => {
    try {
        const tasks = await Task.find();
        res.json(tasks);
    } catch (err) {
        res.status(500).json({ error: 'Failed to fetch tasks' });
    }
});
```

### Code Explanation

- `app.get('/tasks', async (req, res) => {...});`

    - This sets up an Express GET route at `/tasks` to handle client requests for retrieving stored tasks.
    - Async/Await: Makes the route handler asynchronous, enabling `await` for database operations without blocking the server.
    - `req` (Request): Represents the incoming HTTP request, including headers, query parameters, and other details.
    - `res` (Response): Used to send the response back to the client, such as JSON data or status codes.

- `try {...}`

    - This wraps the task retrieval logic in a try block to handle potential errors. It ensures that any errors during database interaction or task retrieval are caught.

- `const tasks = await Task.find();`

    - This line uses the mongoose method `.find()` to retrieve all matching documents. Since no query is specified, it fetches all documents in the `tasks` collection.
    - `await`: Waits for the database query to complete before moving to the next line.

- `res.json(tasks);`

    - This sends the retrieved tasks as a JSON response to the client.

- `catch (err) {...}`

    - This catches any errors that occur during the task retrieval process (for example, database connection issues or invalid data).

- `res.status(500).json({ error: 'Failed to fetch tasks' });`

    - This sends an error response to the client with an HTTP status code of 500 (Internal Server Error). It notifies the client that the task retrieval failed and provides an error message.

## Step 3.3: Implement the Update Operation

The update operation will be used to modify existing data in the database. Here is how to implement it:

```javascript
app.put('/tasks/:id', async (req, res) => {
    const { id } = req.params;
    const { title } = req.body;

    try {
        const updatedTask = await Task.findByIdAndUpdate(id, { title }, { new: true });
        res.json(updatedTask);
    } catch (err) {
        res.status(500).json({ error: 'Failed to update task' });
    }
});
```

### Code Explanation

- `app.put('/tasks/:id', async (req, res) => {...});`

    - This line sets up an Express route to handle HTTP PUT requests at the endpoint /tasks/:id.
    - `:id` is a route parameter that acts as a placeholder for the ID of the task to be updated.

- `const { id } = req.params;`

    - This extracts the id parameter from the URL.
    - Example: For a request to /tasks/12345, id will be '12345'.

- `const { title } = req.body;`

    This extracts the title field from the request body.

- `try {...}`

    - This wraps the task update logic in a try block to handle potential errors. It ensures that any errors during database interaction or task update are caught.

- `const updatedTask = await Task.findByIdAndUpdate(id, { title }, { new: true });`

    - This line used the mongoose method `.findByIdAndUpdate()` to find a document by its id and update it and stores the result in updatedTask.
    - id: The ID of the task to update, extracted from the URL parameter.
    - { title }: The update to apply (in this case, changing the task's title).
    - { new: true }: This ensures the method returns the updated task after applying changes (instead of the old version).

- `res.json(updatedTask);`

    - This sends the updated task as a JSON response to the client.

- `catch (err) {...}`

    - This catches any errors that occur during the task update process (for example, database connection issues or invalid data).

- `res.status(500).json({ error: 'Failed to update task' });`

    - This sends an error response to the client with an HTTP status code of 500 (Internal Server Error). It notifies the client that the task update failed and provides an error message.




## Step 3.4: Implement the Delete Operation

The delete operation will be used to remove data from the database. Here is how to implement it:

```javascript
app.delete('/tasks/:id', async (req, res) => {
    const { id } = req.params;

    try {
        await Task.findByIdAndDelete(id);
        res.json({ message: 'Task deleted successfully' });
    } catch (err) {
        res.status(500).json({ error: 'Failed to delete task' });
    }
});
```

### Code Explanation

- `app.delete('/tasks/:id', async (req, res) => {...});`

    - This line defines a DELETE route for the endpoint /tasks/:id.
    - `:id` is a route parameter that acts as a placeholder for the ID of the task to be deleted.

- `const { id } = req.params;`

    - This extracts the id parameter from the URL.

- `try {...}`

    - This wraps the task deletion logic in a try block to handle potential errors. It ensures that any errors during database interaction or task deletion are caught.

- `await Task.findByIdAndDelete(id);`

    - This line uses the Mongoose `.findByIdAndDelete()` method to locate and delete a task by its id associated with the tasks collection.

- `res.json({ message: 'Task deleted successfully' });`

    - This line sends a JSON response back to the client with a success message.

- `catch (err) {...}`

    - This catches any errors that occur during the task deletion process (for example, database connection issues or invalid data).

- `res.status(500).json({ error: 'Failed to delete task' });`

    - This sends an error response to the client with an HTTP status code of 500 (Internal Server Error). It notifies the client that the task deletion failed and provides an error message.


# Step 4: Specify Port

Finally we can specify the port the server will listen on.

```javascript
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
```

### Code Explanation

- `const PORT = process.env.PORT || 5000;`

    - This sets the PORT variable to the value of process.env.PORT if it exists, or defaults to 5000 if it doesn't. It is retrieved from our .env file.

- `app.listen(PORT, () => console.log(`Server running on port ${PORT}`))`

    - This starts the Express server and listens for incoming requests on the specified PORT.
    - The listen method binds the server to the specified port and starts the event loop, making the server ready to handle requests.
    - The arrow function `() => console.log(...)` is executed once the server successfully starts. It then logs a message to the console indicating which port the server is running on (useful for debugging).

# Step 5: Entire Code

```javascript
// Import Dependencies
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Connect to MongoDB Atlas
mongoose.connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
.then(() => console.log('Connected to MongoDB Atlas'))
.catch((err) => console.error('Error connecting to MongoDB:', err));

// Define Task Schema
const taskSchema = new mongoose.Schema({
    title: { type: String, required: true },
});

// Create Task Model
const Task = mongoose.model('Task', taskSchema);

// Create
app.post('/tasks', async (req, res) => {
    const { title } = req.body;

    try {
        const newTask = new Task({ title });
        await newTask.save();
        res.status(201).json(newTask);
    } catch (err) {
        res.status(500).json({ error: 'Failed to create task' });
    }
});

// Read
app.get('/tasks', async (req, res) => {
    try {
        const tasks = await Task.find();
        res.json(tasks);
    } catch (err) {
        res.status(500).json({ error: 'Failed to fetch tasks' });
    }
});

// Update
app.put('/tasks/:id', async (req, res) => {
    const { id } = req.params;
    const { title } = req.body;

    try {
        const updatedTask = await Task.findByIdAndUpdate(id, { title }, { new: true });
        res.json(updatedTask);
    } catch (err) {
        res.status(500).json({ error: 'Failed to update task' });
    }
});

// Delete
app.delete('/tasks/:id', async (req, res) => {
    const { id } = req.params;

    try {
        await Task.findByIdAndDelete(id);
        res.json({ message: 'Task deleted successfully' });
    } catch (err) {
        res.status(500).json({ error: 'Failed to delete task' });
    }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

```

# Step 6: Testing

We have now finished the backend of our CRUD application. To test what we have made we open the terminal, change to the backend directory (if youre not in it already), and run the server. In terminal this is what you will type:

```terminal
cd backend
npx nodemon index.js
```

Now type http://localhost:5000/tasks into your web browser and you should see something like this:

![Backend Local Host](assets/mern-tutorial/Backend-Finish.jpg)

As there is no frontend to interact with all we see is the data in the database which is currently empty. To test our database and API connect we can use postman.


