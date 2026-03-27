#!/bin/bash
PROJECT_DIR="$HOME/health-tracker"
LOG_FILE="$PROJECT_DIR/bot.log"
PID_FILE="$PROJECT_DIR/.bot.pid"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null
}

start() {
    echo -e "${GREEN}[START] Health Tracker...${NC}"
    cd "$PROJECT_DIR"

    echo "  [1/4] Starting Docker containers..."
    docker compose up -d

    echo "  [2/4] Waiting for Weaviate..."
    for i in $(seq 1 15); do
        if curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
            echo "         Weaviate ready!"; break
        fi
        sleep 2
    done

    if ! pgrep -x "ollama" > /dev/null; then
        echo "  [3/4] Starting Ollama..."
        ollama serve > /dev/null 2>&1 &
        sleep 3
    else
        echo "  [3/4] Ollama already running"
    fi

    eval "$(ssh-agent -s)" > /dev/null 2>&1
    ssh-add ~/.ssh/fitnessbot > /dev/null 2>&1

    > "$LOG_FILE"

    echo "  [4/4] Starting bot..."
    set -a
    source "$PROJECT_DIR/.env"
    set +a
    PYTHONUNBUFFERED=1 /usr/bin/python3 -u "$PROJECT_DIR/main.py" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    echo -e "${GREEN}------------------------------------------${NC}"
    echo -e "${GREEN} Bot is running! Showing live logs below:${NC}"
    echo -e "${GREEN}------------------------------------------${NC}\n"

    tail -f "$LOG_FILE"
}

stop() {
    echo -e "\n${RED}[STOP] Stopping Health Tracker...${NC}"

    if [ -f "$PID_FILE" ]; then
        kill "$(cat $PID_FILE)" 2>/dev/null
        rm "$PID_FILE"
        echo "  [1/3] Bot stopped"
    fi
    pkill -f "main.py" 2>/dev/null

    cd "$PROJECT_DIR"
    docker compose down
    echo "  [2/3] Docker stopped"

    pkill -x ollama 2>/dev/null
    echo "  [3/3] Ollama stopped"

    pkill -f "tail -f $LOG_FILE" 2>/dev/null

    echo -e "\n${GREEN}------------------------------------------${NC}"
    echo -e "${GREEN} All stopped. Close window with X.${NC}"
    echo -e "${GREEN}------------------------------------------${NC}"
}

case "${1:-toggle}" in
    start)  start ;;
    stop)   stop ;;
    toggle)
        if is_running; then
            stop
        else
            start
        fi
        ;;
esac