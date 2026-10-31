#!/usr/bin/env bash
# SpinSlicer launcher.
#
#   bash spinslicer.sh           start the app (sets everything up on the first run)
#   bash spinslicer.sh --reset   rebuild the private Python environment, then start
#   bash spinslicer.sh --install-shortcut   put a SpinSlicer icon in the menu and on the desktop
#
# The first run creates .venv next to this file in a terminal window so the
# progress is visible (3-10 minutes, needs internet). Later runs start at once.
# App output goes to ~/.cache/spinslicer/launcher.log; a crash shows a dialog.
set -u

APP_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$APP_DIR" || exit 1
STATE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/spinslicer"
LOG="$STATE_DIR/launcher.log"
STAMP="$STATE_DIR/python"
ICON="$APP_DIR/assets/spinslicer.svg"
VENV="$APP_DIR/.venv"
mkdir -p "$STATE_DIR"
REQ_HASH="$(sha256sum requirements.txt 2>/dev/null | cut -c1-16)"
CHECK="import PyQt6.QtWidgets, pyvistaqt, trimesh, scipy, skimage, PIL, cv2, matplotlib"

notify() {
    command -v notify-send >/dev/null 2>&1 && notify-send -a SpinSlicer -i "$ICON" SpinSlicer "$1"
    return 0
}

error_box() {
    if command -v kdialog >/dev/null 2>&1; then
        kdialog --title SpinSlicer --icon "$ICON" --error "$1"
    elif command -v zenity >/dev/null 2>&1; then
        zenity --error --title=SpinSlicer --no-markup --text="$1"
    else
        notify "$1"
    fi
}

works() { [ -n "$1" ] && [ -x "$1" ] && "$1" -c "$CHECK" >/dev/null 2>&1; }

remember() { printf '%s %s\n' "$REQ_HASH" "$1" > "$STAMP"; }

# Prints a Python that can run SpinSlicer, or fails.
find_python() {
    local hash py
    if [ -f "$STAMP" ]; then
        read -r hash py < "$STAMP"
        if [ "$hash" = "$REQ_HASH" ] && [ -x "$py" ]; then
            echo "$py"
            return 0
        fi
    fi
    # requirements.txt changed (an update): top up our own .venv quietly, keep going if offline.
    if [ -x "$VENV/bin/python" ] && [ -f "$STAMP" ]; then
        notify "Обновление: доустанавливаю новые библиотеки…" >&2
        "$VENV/bin/python" -m pip install -q -r requirements.txt >> "$STATE_DIR/update.log" 2>&1 || true
    fi
    for py in "$VENV/bin/python" "$(command -v python3 || true)"; do
        if works "$py"; then
            remember "$py"
            echo "$py"
            return 0
        fi
    done
    return 1
}

# One base interpreter -> fresh .venv with all requirements. Pinned versions
# first; if a pin has no wheel for this Python, the newest compatible versions.
build_venv() {
    local base="$1"
    echo
    echo "==> Python: $base ($("$base" -c 'import sys; print(sys.version.split()[0])'))"
    rm -rf "$VENV"
    "$base" -m venv "$VENV" || return 1
    "$VENV/bin/python" -m pip install --upgrade pip wheel || return 1
    if ! "$VENV/bin/python" -m pip install -r requirements.txt; then
        echo
        echo "==> Точные версии не встали, ставлю совместимые новые."
        sed -E 's/==[^ ;#]+//' requirements.txt > "$STATE_DIR/requirements-loose.txt"
        "$VENV/bin/python" -m pip install -r "$STATE_DIR/requirements-loose.txt" || return 1
    fi
    works "$VENV/bin/python"
}

setup() {
    echo "SpinSlicer — первая настройка. Это один раз, 3–10 минут, нужен интернет."
    local base real tried=" " ok=1
    # Supported 3.11/3.12 first, then whatever the system and conda have.
    for base in "$(command -v python3.12 || true)" "$(command -v python3.11 || true)" \
                "$(command -v python3 || true)" "$HOME/miniconda3/bin/python3" \
                "$(command -v python3.13 || true)"; do
        [ -n "$base" ] && [ -x "$base" ] || continue
        real="$(readlink -f "$base")"
        case "$tried" in *" $real "*) continue ;; esac
        tried="$tried$real "
        if build_venv "$base"; then
            ok=0
            break
        fi
        echo "==> С этим Python не вышло, пробую следующий."
    done
    if [ $ok -ne 0 ]; then
        rm -rf "$VENV"
        echo
        echo "Не удалось подготовить окружение. Проверь интернет и что установлен python."
        return 1
    fi
    # Optional C++ SIRT backend (square cuvette only); the app works without it.
    "$VENV/bin/python" build_sirt.py >/dev/null 2>&1 || echo "(C++ SIRT не собран — не страшно)"
    remember "$VENV/bin/python"
    echo
    echo "Готово! Запускаю SpinSlicer…"
    return 0
}

# Runs setup in a visible terminal when one exists, otherwise in the background.
setup_in_terminal() {
    local self="$APP_DIR/spinslicer.sh" term
    for term in konsole gnome-terminal kitty alacritty xfce4-terminal xterm; do
        command -v "$term" >/dev/null 2>&1 || continue
        case "$term" in
            gnome-terminal) gnome-terminal --wait -- bash "$self" --setup ;;
            kitty)          kitty bash "$self" --setup ;;
            *)              "$term" -e bash "$self" --setup ;;
        esac
        return
    done
    notify "Первая настройка: ставлю библиотеки (3–10 минут)…"
    setup > "$STATE_DIR/setup.log" 2>&1 || {
        error_box "Не удалось подготовить окружение. Подробности: $STATE_DIR/setup.log"
        exit 1
    }
}

# Desktop entry pointing at this copy of the app (menu or desktop icon).
write_entry() {
    local file="$1"
    mkdir -p "$(dirname "$file")"
    cat > "$file" <<EOF
[Desktop Entry]
Type=Application
Name=SpinSlicer
GenericName=Slicer for volumetric printing
GenericName[ru]=Слайсер для объёмной печати
Comment=Slice models into projections for tomographic (CAL/VAM) printing
Comment[ru]=Нарезка моделей на проекции для томографической (CAL/VAM) печати
Exec=bash "$APP_DIR/spinslicer.sh"
Path=$APP_DIR
Icon=$ICON
Terminal=false
StartupNotify=true
StartupWMClass=SpinSlicer
Categories=Science;Graphics;Engineering;
Keywords=slicer;3d;print;cal;vam;tomographic;
Actions=reset;

[Desktop Action reset]
Name=Rebuild environment
Name[ru]=Переустановить окружение
Exec=bash "$APP_DIR/spinslicer.sh" --reset
EOF
    chmod +x "$file"
    # GNOME asks to trust desktop launchers; KDE trusts executable ones.
    command -v gio >/dev/null 2>&1 && gio set "$file" metadata::trusted true 2>/dev/null
    return 0
}

MENU_ENTRY="${XDG_DATA_HOME:-$HOME/.local/share}/applications/spinslicer.desktop"

# Keeps the menu entry pointing here, even after the folder is moved.
ensure_menu_entry() {
    grep -qF "Exec=bash \"$APP_DIR/spinslicer.sh\"" "$MENU_ENTRY" 2>/dev/null || write_entry "$MENU_ENTRY"
    local desk
    desk="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")/SpinSlicer.desktop"
    [ -f "$desk" ] && [ ! -x "$desk" ] && chmod +x "$desk"
    return 0
}

launch() {
    local py="$1" code
    { echo "=== $(date '+%F %T') $py"; } > "$LOG"
    "$py" "$APP_DIR/SpinSlicer.py" >> "$LOG" 2>&1
    code=$?
    if [ $code -ne 0 ]; then
        rm -f "$STAMP"          # re-check the environment on the next start
        error_box "$(printf 'SpinSlicer закрылся с ошибкой (код %s).\n\n%s\n\nЖурнал: %s' \
            "$code" "$(tail -n 12 "$LOG")" "$LOG")"
    fi
    return $code
}

case "${1:-}" in
    --setup)
        # Inside the terminal window opened by setup_in_terminal.
        if setup; then
            setsid bash "$APP_DIR/spinslicer.sh" >/dev/null 2>&1 < /dev/null &
            sleep 2
        else
            read -rp "Нажми Enter, чтобы закрыть окно."
        fi
        exit 0
        ;;
    --reset)
        rm -f "$STAMP"
        rm -rf "$VENV"
        ;;
    --install-shortcut)
        write_entry "$MENU_ENTRY"
        desk="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
        write_entry "$desk/SpinSlicer.desktop"
        echo "Готово: SpinSlicer в меню приложений и на рабочем столе ($desk)."
        exit 0
        ;;
esac

ensure_menu_entry

if PY="$(find_python)"; then
    launch "$PY"
    exit $?
fi
setup_in_terminal
# The terminal path starts the app itself; the background path continues here.
if PY="$(find_python)"; then
    pgrep -f "$APP_DIR/SpinSlicer.py" >/dev/null 2>&1 || launch "$PY"
fi
