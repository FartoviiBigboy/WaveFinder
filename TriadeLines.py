from CustomInfiniteLine import CustomInfiniteLine


class VerticalTriadeLine:
    def __init__(self, pos=0):
        self.base_pen: tuple[int, int, int] = (0, 0, 0)
        self.hover_pen: tuple[int, int, int] = (255, 190, 11)
        self.lines: list[CustomInfiniteLine] = [
            CustomInfiniteLine(parent=self, angle=90, pen=self.base_pen, hoverPen=self.hover_pen, pos=pos)
            for _ in range(3)
        ]

    def connectClick(self, method) -> None:
        for line in self.lines:
            line.sigClicked.connect(method)

    def disconnectPositionFinished(self) -> None:
        for line in self.lines:
            line.sigPositionChangeFinished.disconnect()

    def connectPositionFinished(self, method) -> None:
        for line in self.lines:
            line.sigPositionChangeFinished.connect(method)

    def setMovables(self, state) -> None:
        for line in self.lines:
            line.setMovable(state)

    def setPos(self, new_pos) -> None:
        for line in self.lines:
            line.setPos(new_pos)

    def setPen(self, pen) -> None:
        for line in self.lines:
            line.setPen(pen)

    def pos(self) -> float:
        return self.lines[0].pos().x()

    def movable(self) -> bool:
        return self.lines[0].movable


class PVerticalTriadeLine(VerticalTriadeLine):
    def __init__(self, pos=0) -> None:
        super().__init__(pos=pos)
        self.base_pen: tuple[int, int, int] = (255, 0, 0)
        for line in self.lines:
            line.setPen(self.base_pen)


class SVerticalTriadeLine(VerticalTriadeLine):
    def __init__(self, pos=0) -> None:
        super().__init__(pos=pos)
        self.base_pen: tuple[int, int, int] = (0, 255, 0)
        for line in self.lines:
            line.setPen(self.base_pen)
