import curses


def mainMenu(stdscr, selection):
    stdscr.clear()
    items = ["Train model", "Create new model", "Delete model", "Exit"]

    for idx, option in enumerate(items):
        if idx == selection:
            stdscr.addstr(idx, 0, option, curses.color_pair(1))
        else:
            stdscr.addstr(idx, 0, option)

    stdscr.refresh()


def main(stdscr):
    curses.curs_set(0)
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    selection = 0

    mainMenu(stdscr, selection)

    while True:

        key = stdscr.getch()

        if key == curses.KEY_UP and selection > 0:
            selection -= 1
        elif key == curses.KEY_DOWN and selection < 3:
            selection += 1
        elif key == 10:
            if selection == 3:
                break
            else:
                stdscr.addstr(5, 0, "You selected: " + str(selection))
                stdscr.refresh()
                stdscr.getch()
        mainMenu(stdscr, selection)


if __name__ == "__main__":
    curses.wrapper(main)
