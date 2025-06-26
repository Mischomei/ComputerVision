import curses

class CursesHandler:
    def __enter__(self):
        self.stdsrc = curses.initscr()
        curses.cbreak()
        curses.noecho()
        self.stdsrc.keypad(1)
        SCREEN_HEIGHT, SCREEN_WIDTH = self.stdsrc.getmaxyx()
        return self.stdsrc

    def __exit__(self, exc_type, exc_val, exc_tb):
        curses.nocbreak()
        self.stdsrc.keypad(0)
        curses.echo()
        curses.endwin()