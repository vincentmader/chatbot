function postAjaxCall(url, data) {
  // return a new promise.
  return new Promise(function (resolve, reject) {
    // do the usual XHR stuff
    var req = new XMLHttpRequest();
    req.open("post", url);
    //NOW WE TELL THE SERVER WHAT FORMAT OF POST REQUEST WE ARE MAKING
    req.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    req.onload = function () {
      if (req.status == 200) {
        resolve(req.response);
      } else {
        reject(Error(req.statusText));
      }
    };
    // handle network errors
    req.onerror = function () {
      reject(Error("Network Error"));
    }; // make the request
    req.send(data);
    //same thing if i hardcode like //req.send("limit=2");
  });
}

let customerID = 1;
let chatID = 1;

const css = document.createElement("link");
const chatButton = document.createElement("div");
const chatWindow = document.createElement("div");
const inputForm = document.createElement("input");
const header = document.createElement("div");
const sendButton = document.createElement("button");

chatButton.id = "chatButton";
chatButton.style.color = "red";
chatButton.style.backgroundColor = "white";
chatButton.style.width = "100px";
chatButton.style.height = "100px";
chatButton.style.position = "absolute";
chatButton.style.right = "10px";
chatButton.style.bottom = "10px";
chatButton.style.visibility = "visible";
chatButton.style.borderRadius = "50%";
chatButton.style.border = "3px solid gray";

chatWindow.id = "chatWindow";
chatWindow.style.backgroundColor = "white";
chatWindow.style.visibility = "hidden";
chatWindow.style.width = "300px";
chatWindow.style.height = "500px";
chatWindow.style.position = "absolute";
chatWindow.style.right = "10px";
chatWindow.style.bottom = "120px";
chatWindow.style.overflow = "hidden";
chatWindow.style.border = "3px solid gray";
chatWindow.style.borderRadius = "10px";
let chatWindowActive = false;
function foo() {
  if (chatWindowActive) {
    chatWindow.style.visibility = "hidden";
    chatWindowActive = false;
  } else {
    chatWindow.style.visibility = "visible";
    chatWindowActive = true;
  }
}
chatButton.addEventListener("click", foo);

inputForm.name = "inputForm";
inputForm.style.position = "absolute";
inputForm.style.right = "0px";
inputForm.style.bottom = "40px";
inputForm.style.width = "100%";
inputForm.style.height = "80px";
inputForm.type = "text";
inputForm.style.border = "0px solid gray";
inputForm.style.borderTop = "1px solid gray";
inputForm.style.boxSizing = "border-box";
inputForm.style.paddingRight = "20px";
inputForm.style.paddingLeft = "20px";
inputForm.style.fontSize = "30px";

header.style.position = "absolute";
header.style.right = "0px";
header.style.borderBottom = "1px solid gray";
header.style.backgroundColor = "white";
header.style.top = "0px";
header.style.width = "100%";
// header.style.paddingRight = "20px";
// header.style.paddingLeft = "20px";
header.style.height = "40px";
header.innerHTML = '<p style="margin-left:20px; font-family: Arial">Chat</p>';

// css.href = "localhost/static/css/style.css";
// css.type = "text/css";
// css.rel = "stylesheet";

sendButtonOnClick = function () {
  postAjaxCall(
    // "https://webhook.site/b501642c-11cd-4329-8973-6f7916f677f6",
    "http://localhost:5001/" + customerID + "/" + chatID,
    inputForm.value
  );
};
sendButton.type = "button";
sendButton.id = "sendButton";
sendButton.value = "Send";
sendButton.style.width = "100%";
// sendButton.style.margin = "0px";
sendButton.style.height = "40px";
sendButton.style.border = "0px";
sendButton.style.borderRadius = "5px";
sendButton.style.backgroundColor = "coral";
sendButton.style.position = "absolute";
// sendButton.style.right = "10px";
sendButton.style.bottom = "0px";
sendButton.onclick = sendButtonOnClick;

// document.head.append(css);
document.body.appendChild(chatWindow);
document.body.appendChild(chatButton);
document.getElementById("chatWindow").appendChild(header);
document.getElementById("chatWindow").appendChild(inputForm);
document.getElementById("chatWindow").appendChild(sendButton);
// document.body.appendChild(jquery);
