// Loads CrisisBench test tweets from the demo server and draws them as a
// plain timeline. This file does no classification at all; the Chrome
// extension finds these tweets on the page and adds the triage labels.

const PAGE_SIZE = 25;

const feedState = {
  mode: "all",      // "all" or "errors"
  event: "all",     // a CrisisBench event name, or "all"
  nextOffset: 0,
  total: 0,
  loading: false,
};

const timelineElement = document.getElementById("timeline-tweets");
const eventPicker = document.getElementById("event-picker");
const loadMoreButton = document.getElementById("load-more");
const messageElement = document.getElementById("feed-message");
const countElement = document.getElementById("feed-count");
const tabButtons = document.querySelectorAll(".feed-tab");


function showMessage(text) {
  messageElement.textContent = text;
  messageElement.hidden = false;
}

function hideMessage() {
  messageElement.hidden = true;
}

function updateCount() {
  let shown = timelineElement.children.length;
  countElement.textContent = shown + " of " + feedState.total + " tweets";
}


// Builds one tweet. Gold labels go into data- attributes so the extension's
// answer key can read them; nothing on the page displays them.
function buildTweetElement(tweet) {
  let article = document.createElement("article");
  article.className = "tweet";
  article.dataset.tweetId = tweet.id;
  article.dataset.goldEndToEnd = tweet.gold_end_to_end;
  article.dataset.goldDisplay = tweet.gold_display;
  article.dataset.goldInformativeness = tweet.gold_informativeness;
  article.dataset.goldHumanitarian = tweet.gold_humanitarian;

  let avatar = document.createElement("div");
  avatar.className = "tweet-avatar";
  avatar.setAttribute("aria-hidden", "true");

  let body = document.createElement("div");
  body.className = "tweet-body";

  let meta = document.createElement("div");
  meta.className = "tweet-meta";

  let name = document.createElement("span");
  name.className = "tweet-name";
  name.textContent = "Anonymous user";

  let eventName = document.createElement("span");
  eventName.className = "tweet-event";
  eventName.textContent = tweet.event_display;

  meta.appendChild(name);
  meta.appendChild(eventName);

  // textContent (not innerHTML) so tweet text can never inject markup.
  let text = document.createElement("p");
  text.className = "tweet-text";
  text.textContent = tweet.text;

  let source = document.createElement("p");
  source.className = "tweet-source";
  source.textContent = "From the " + tweet.source + " collection";

  body.appendChild(meta);
  body.appendChild(text);
  body.appendChild(source);

  article.appendChild(avatar);
  article.appendChild(body);
  return article;
}


async function loadEvents() {
  let response = await fetch("/api/events");
  if (!response.ok) {
    return;
  }
  let data = await response.json();

  for (let item of data.events) {
    let option = document.createElement("option");
    option.value = item.event;
    option.textContent = item.display + " (" + item.disaster_type + ", " + item.count + ")";
    eventPicker.appendChild(option);
  }
}


async function checkKnownMistakes() {
  let response = await fetch("/api/status");
  if (!response.ok) {
    return;
  }
  let status = await response.json();
  if (status.known_mistakes_available) {
    return;
  }

  for (let button of tabButtons) {
    if (button.dataset.mode === "errors") {
      button.disabled = true;
      button.title = "Start the server with --errors to turn this on";
    }
  }
}


async function loadNextPage() {
  if (feedState.loading) {
    return;
  }
  feedState.loading = true;
  loadMoreButton.hidden = true;
  hideMessage();

  let url = "/api/feed?event=" + encodeURIComponent(feedState.event)
    + "&mode=" + encodeURIComponent(feedState.mode)
    + "&offset=" + feedState.nextOffset
    + "&limit=" + PAGE_SIZE;

  let response;
  try {
    response = await fetch(url);
  } catch (error) {
    showMessage("Can't reach the demo server. Start it with python server/app.py, then reload.");
    feedState.loading = false;
    return;
  }

  if (!response.ok) {
    let problem = await response.json();
    showMessage(problem.detail);
    feedState.loading = false;
    return;
  }

  let data = await response.json();
  for (let tweet of data.tweets) {
    let element = buildTweetElement(tweet);
    timelineElement.appendChild(element);
  }

  feedState.nextOffset = data.next_offset;
  feedState.total = data.total;
  updateCount();

  if (data.total === 0) {
    showMessage("No test tweets match this view. Pick another disaster.");
  }
  if (feedState.nextOffset < feedState.total) {
    loadMoreButton.hidden = false;
  }
  feedState.loading = false;
}


function resetFeed() {
  timelineElement.replaceChildren();
  feedState.nextOffset = 0;
  feedState.total = 0;
  loadNextPage();
}


function selectTab(selectedButton) {
  for (let button of tabButtons) {
    let isSelected = button === selectedButton;
    button.classList.toggle("is-selected", isSelected);
    button.setAttribute("aria-selected", String(isSelected));
  }
  feedState.mode = selectedButton.dataset.mode;
  resetFeed();
}


for (let button of tabButtons) {
  button.addEventListener("click", function () {
    selectTab(button);
  });
}

eventPicker.addEventListener("change", function () {
  feedState.event = eventPicker.value;
  resetFeed();
});

loadMoreButton.addEventListener("click", function () {
  loadNextPage();
});


async function startFeed() {
  await loadEvents();
  await checkKnownMistakes();
  loadNextPage();
}

startFeed();
