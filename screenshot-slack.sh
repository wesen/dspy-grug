#!/bin/bash

# Check if slack-context.json exists
if [ ! -f "slack-context.json" ]; then
    echo "Please authenticate in the browser..."
    shot-scraper auth 'https://app.slack.com/block-kit-builder/TCSEMFMAP#%7B%22blocks%22:%5B%7B%22type%22:%22divider%22%7D%5D%7D' slack-context.json
    exit 1
fi

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <json_file or - for stdin>"
    exit 1
fi

# Read JSON input
if [ "$1" == "-" ]; then
    json_input=$(cat)
else
    json_input=$(cat "$1")
fi

# Encode JSON for URL
encoded_json=$(echo "$json_input" | jq -c | jq -sRr @uri)

echo "Running shot-scraper command, this can take a little moment..."
# Run shot-scraper command
shot-scraper -o /tmp/foo2.png -a slack-context.json \
    "https://app.slack.com/block-kit-builder/TCSEMFMAP#$encoded_json" \
    -s .p-bkb_preview_container \
    --wait-for "document.querySelectorAll('.p-bkb_preview_container')" \
    --bypass-csp

# Open the generated image
xdg-open /tmp/foo2.png
