// client/script.js

const chatBox = document.getElementById("sentiment-box");

async function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    if (!userInput) return;

    addMessage(userInput, "user");
    document.getElementById("user-input").value = "";

    try {
        const response = await fetch("http://127.0.0.1:8000/sentiment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ "text": userInput })
        });

        if (!response.ok) {
            throw new Error("Network response was not ok.");
        }

        const data = await response.json();
        console.log("Response data:", data);  // Log the response data

        // Check if data contains the expected fields
        if (data.sentiment !== undefined && data.confidence !== undefined) {
            //addMessage(`Sentiment: ${data.sentiment}, Confidence: ${data.confidence}`, "bot");
            //Just specify the emotion alone
            addMessage(`Sentiment: ${data.sentiment}`, "bot");
        } else {
            addMessage("Error: Invalid response from server.", "bot");
        }
    } catch (error) {
        console.error("Error:", error);
        addMessage("Error: Could not reach the server.", "bot");
    }
}


function addMessage(message, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.className = "sentiment-message";
    messageDiv.classList.add(sender === "user" ? "user" : "bot");
    messageDiv.textContent = (sender === "user" ? "You: " : "Bot: ") + message;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}
