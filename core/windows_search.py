from __future__ import annotations

import time
import pyautogui
import pyperclip

from .debug import debug_event


# from .window_manager import get_active_window_info # Можно оставить, если нужно

def windows_search_open(query: str) -> None:
    # 1. СБРОС "ЗАЛИПШИХ" КЛАВИШ (Исправлено на keyDown/keyUp)
    try:
        pyautogui.keyUp('win')
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')
    except:
        pass

    for attempt in range(1, 3):
        debug_event("APP_OPEN", f"strategy=windows_search attempt={attempt}")
        try:
            # Копируем текст
            pyperclip.copy(query)

            # 2. ОТКРЫТИЕ МЕНЮ (Используем верный регистр: keyDown)
            pyautogui.keyDown('ctrl')
            pyautogui.press('esc')
            pyautogui.keyUp('ctrl')

            # Ждем появления меню
            time.sleep(0.2)

            # 3. ВСТАВКА (Shift + Insert)
            pyautogui.keyDown('shift')
            time.sleep(0.1)
            pyautogui.press('insert')
            time.sleep(0.1)
            pyautogui.keyUp('shift')

            # Ждем, пока поиск отобразит результаты
            time.sleep(0.2)

            # 4. ВВОД
            pyautogui.press("enter")

            time.sleep(0.5)
            print(f"[APP_OPEN] attempt={attempt} ok=true")
            break

        except Exception as exc:
            print(f"[APP_OPEN] attempt={attempt} ok=false error={exc}")
            pyautogui.press("esc")
            time.sleep(0.3)