// VARIABLE DEFINITIONS
// ====================

let customerID = 1;
let chatID = 1;

const chatButton = document.createElement("div");
const chatWindow = document.createElement("div");
const inputForm = document.createElement("textarea");
const header = document.createElement("div");
const sendButton = document.createElement("button");
const messageList = document.createElement("div");
let chatWindowActive = false;

// FUNCTION DEFINITIONS
// ====================

// get & post requests
function httpPost(url, data) {
  // return a new promise.
  return new Promise(function (resolve, reject) {
    // do the usual XHR stuff
    var req = new XMLHttpRequest();
    req.open("post", url);
    //NOW WE TELL THE SERVER WHAT FORMAT OF POST REQUEST WE ARE MAKING
    req.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    req.onload = function () {
      if (req.status === 200) {
        resolve(req.response);
      } else {
        reject(Error(req.statusText));
      }
    };
    // handle network errors
    //
    req.onerror = function () {
      reject(Error("Network Error"));
    }; // make the request
    req.send(data);
    //same thing if i hardcode like //req.send("limit=2");
  });
}

function httpGet(url, callback) {
  let xmlHttp = new XMLHttpRequest();
  xmlHttp.onreadystatechange = function () {
    if (xmlHttp.readyState === 4 && xmlHttp.status === 200)
      callback(xmlHttp.responseText);
  };
  xmlHttp.open("GET", url, true); // true for asynchronous
  xmlHttp.send(null);
}

// button onClick functions
function chatWindowShowButtonOnClick() {
  if (chatWindowActive) {
    chatWindow.style.visibility = "hidden";
    chatWindowActive = false;
  } else {
    chatWindow.style.visibility = "visible";
    chatWindowActive = true;
  }
}

function sendButtonOnClick() {
  httpPost(
    // "https://webhook.site/b501642c-11cd-4329-8973-6f7916f677f6",
    "http://localhost:5001/" + customerID + "/" + chatID,
    inputForm.value
  );
  inputForm.value = ""; // clear input
  httpGet("http://localhost:5001/messageList", createMsgList);
  setTimeout(
    () => httpGet("http://localhost:5001/messageList", createMsgList),
    1000
  );
}

// create html elements
function createMsgList(res) {
  let messages = JSON.parse(res)["messages"];
  document.getElementById("messageList").outerHTML = "";
  const messageList = document.createElement("div");
  messageList.id = "messageList";
  messageList.style.overflow = "scroll";
  messageList.style.position = "absolute";
  messageList.style.height = "350px";
  messageList.style.top = "40px";
  messageList.style.padding = "5px";
  document.getElementById("chatWindow").appendChild(messageList);
  for (let idx = 0; idx < messages.length; idx++) {
    msg = messages[idx];

    const msgBubble = document.createElement("div");
    msgBubble.id = "msgBubble";
    msgBubble.style.width = "200px";
    msgBubble.style.border = "2px solid green";
    msgBubble.style.borderRadius = "20px";
    msgBubble.style.color = "black";
    msgBubble.style.margin = "5px";
    document.getElementById("messageList").appendChild(msgBubble);

    if (msg["sender"] !== "bot") {
      msgBubble.style.marginLeft = "80px";
      msgBubble.style.backgroundColor = "green";
      msgBubble.style.color = "white";
    }

    msgBubble.innerHTML =
      '<p style="margin-left:20px; font-family: Arial,sans-serif">' +
      convertDateToStr(Number(msg["timestamp"])) +
      "<br/><br/>" +
      msg["msg_content"] +
      "</p>";
  }
}

// define IDs for HTML elements
function defineHTMLIDs() {
  chatButton.id = "chatButton";
  chatWindow.id = "chatWindow";
  sendButton.id = "sendButton";
  messageList.id = "messageList";
}

// append HTML elements to document
function appendElementsToDoc() {
  document.body.appendChild(chatWindow);
  document.body.appendChild(chatButton);
  document.getElementById("chatWindow").appendChild(messageList);
  document.getElementById("chatWindow").appendChild(header);
  document.getElementById("chatWindow").appendChild(inputForm);
  document.getElementById("chatWindow").appendChild(sendButton);
}

// styles
function initializeStyles() {
  chatButton.style.color = "red";
  chatButton.style.backgroundColor = "white";
  chatButton.style.width = "70px";
  chatButton.style.height = "70px";
  chatButton.style.position = "absolute";
  chatButton.style.right = "10px";
  chatButton.style.bottom = "10px";
  chatButton.style.visibility = "visible";
  chatButton.style.borderRadius = "50%";
  chatButton.style.border = "3px solid gray";
  chatButton.addEventListener("click", chatWindowShowButtonOnClick);

  chatWindow.style.backgroundColor = "white";
  chatWindow.style.visibility = "hidden";
  chatWindow.style.width = "300px";
  chatWindow.style.height = "500px";
  chatWindow.style.position = "absolute";
  chatWindow.style.right = "10px";
  chatWindow.style.bottom = "100px";
  chatWindow.style.overflow = "hidden";
  chatWindow.style.border = "3px solid gray";
  chatWindow.style.borderRadius = "10px";

  inputForm.name = "inputForm";
  inputForm.style.position = "absolute";
  inputForm.style.right = "0px";
  inputForm.style.bottom = "40px";
  inputForm.style.width = "100%";
  inputForm.style.height = "60px";
  inputForm.type = "text";
  inputForm.style.border = "0px solid gray";
  inputForm.style.borderTop = "1px solid gray";
  inputForm.style.boxSizing = "border-box";
  inputForm.style.paddingRight = "20px";
  inputForm.style.paddingLeft = "20px";
  inputForm.style.fontSize = "20px";
  inputForm.style.fontFamily = "Arial,sans-serif";

  header.style.position = "absolute";
  header.style.right = "0px";
  header.style.borderBottom = "1px solid gray";
  header.style.backgroundColor = "white";
  header.style.top = "0px";
  header.style.width = "100%";
  header.style.height = "40px";
  header.innerHTML =
    '<p style="margin-left:20px; font-family: Arial,sans-serif">Chat</p>';

  sendButton.type = "button";
  sendButton.innerHTML = "Send";
  sendButton.style.color = "white";
  sendButton.style.width = "100%";
  sendButton.style.height = "40px";
  sendButton.style.border = "0px";
  sendButton.style.borderRadius = "5px";
  sendButton.style.backgroundColor = "green";
  sendButton.style.position = "absolute";
  sendButton.style.bottom = "0px";
  sendButton.onclick = sendButtonOnClick;
}

// various
function convertIntToMonthName(num) {
  monthNames = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
  };
  return monthNames[num];
}

function convertDateToStr(timestamp) {
  // python dt timestamps are in secs,
  // js timestamps are in millisec -> factor 1000
  date = new Date(timestamp * 1000);
  return (
    // String(date.getFullYear()) +
    convertIntToMonthName(date.getMonth() + 1) +
    " " +
    // "-" +
    intToDoubleDigit(date.getDate()) +
    ". at " +
    intToDoubleDigit(date.getHours()) +
    ":" +
    intToDoubleDigit(date.getMinutes())
  );
}

function intToDoubleDigit(num) {
  if (num < 10) {
    return "0" + String(num);
  } else {
    return String(num);
  }
}

// MAIN
// ====

function main() {
  // create HTML elements
  defineHTMLIDs();
  appendElementsToDoc();
  initializeStyles();

  // initialize messageList
  // while (true) {
  //   var _ = setInterval(
  httpGet("http://localhost:5001/messageList", createMsgList);
  // 10000
  // );
  // }
}

main();
