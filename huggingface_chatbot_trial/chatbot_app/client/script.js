// client/script.js

const chatBox = document.getElementById("chat-box");

async function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    if (!userInput) return;

    addMessage(userInput, "user");
    document.getElementById("user-input").value = "";

    try {
        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ "text": userInput })
        });
        const data = await response.json();
        addMessage(data.response, "bot");
    } catch (error) {
        console.error("Error:", error);
        addMessage("Error: Could not reach the server.", "bot");
    }
}

function addMessage(message, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.className = "chat-message";
    messageDiv.classList.add(sender === "user" ? "user" : "bot");
    messageDiv.textContent = (sender === "user" ? "You: " : "Bot: ") + message;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}
