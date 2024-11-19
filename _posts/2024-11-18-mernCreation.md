---
title: MERN Tutorial 1 - Initialisation
date: 2024-11-18 18:00:00 +0000
categories: [MERN Tutorial]
tags: [MERN, CRUD, Website Tutorial, Backend, Frontend, ]
math: true
---

# Introduction

Welcome to the MERN Tutorial series. In this series, we will create a basic Create, Read, Update, and Delete application using the MERN stack (MongoDB, Express, React, Node). I will explain each step of this tutorial, breaking down all code, techniques, and software used. At the end of this tuturial we will also upload our application to the internet for free using existing applications. In this blog, we will learn how to initialise our MERN project. Let us begin. 

## Step 1: Open Visual Studio Code and Navigate to Your Project Directory

The first step in any application tutorial is of course opening your IDE (Integrated Development Environment). This is where you will write your code for the application. I am using and recommend Visual Studio Code for its ease of use.

![VS Code Home](assets/mern-tutorial/Home-1.jpg)

Once within VS Code you will be met with the home screen as shown above. Once here you will want to navigate to your project directory. This is where you will store and save your project. You can do this by clicking Open Folder and choosing the folder you want to store your application in.

Now that you are in your project directory we want to make a new directory (folder) to store our application in. You can do this by opening the terminal, type mkdir and then the name of your new directory:

```terminal
mkdir Mern-App-2
```

Now that we have our new directory we want to navigate to it by typing cd then our directory name in the terminal:

```terminal
cd Mern-App-2
```

Now we have a directory for our application we can now start making our project.

## Step 2: Initialise a New Node Project

Once were in our new directory we can start our new project we must first create a new Node project. We do this by typing npm init -y in the terminal:

```terminal
npm init -y
```

This will initialise a new node project and create a file named package.json.

## Step 4: Install Necessary Dependencies

We now need to install the necessary dependencies for our application. As we are using the MERN stack we need to install mongoose and express in addition to cors and dotenv. In the terminal type:

```terminal
npm install mongoose express cors dotenv
```

## Step 5: Install Development Dependencies

We now need to install Nodemon which is a tool that automatically restarts your Node.js server whenever you make changes to your code, saving you from manually restarting it. 

```terminal
npm install --save-dev nodemon
```

Running npm install --save-dev nodemon installs it as a development dependency, ensuring it's only used during development and not in production.


## Step 6: Create a Github Repository

Before continuing with our application, we must set up source control to save our work to GitHub and make commits as needed. It's best practice to create the GitHub repository immediately after setting up the project directory to track all changes from the start. To do this we type the following in the terminal:

```terminal
git init
```

This will initialise our github repository. Now a handy technique with VS Code is that you dont have to use the terminal for source control. Instead you can go to the source control area by pressing Ctrl+Shift+G. here you can click publish branch, give your repo a name, and decide whether to make it a public or private repository. You can then make your initial commit and publish the branch.

## Step 7: Build the Backend

Now we have the necessary dependencies installed we can start structuring our project. First we need to create a new folder named backend. In the terminal type:

```terminal
mkdir backend
```

## Step 8: Create index.js File

Now you can make a new file in the backend directory called index.js to start the express server. For now we will keep it empty and go on to structuring our frontend.

## Step 9: Build the Frontend

Before we build the frontend ensure you are in the root directory. To go back a folder in terminal type:

```terminal
cd ..
```

Ensure youre in the root directory then type:

```terminal
npx create-react-app frontend
```

This will create our react project in a folder named frontend.

## Step 10: .gitignore .env files

Now that we have our structure set up we can create a .gitignore and .env file

In the .gitignore file add:

```
node_modules/
.env
```

This will ensure these files are not added to the github repository, keeping sensitive information safe and removing redundant dependencies from the repository.

# Conclusion

Well done! We have now successfully initialised our MERN project. Proceed to MERN Tutorial 2 for guidance on building the backend and integrating MongoDB.
