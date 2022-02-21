import parse


class DetBbox:
    format_str = "({:0.3f}, {:0.3f}, {:0.3f}, {:0.3f})"

    def __init__(self, xc, yc, w, h):
        self.xc = xc
        self.yc = yc
        self.w = w
        self.h = h

    @property
    def xmin(self):
        return self.xc - self.w / 2

    @property
    def xmax(self):
        return self.xc + self.w / 2

    @property
    def ymin(self):
        return self.yc - self.h / 2

    @property
    def ymax(self):
        return self.yc + self.h / 2

    def __str__(self) -> str:
        return self.format_str.format(self.xc, self.yc, self.w, self.h)

    @classmethod
    def from_string(cls, bbox_str):
        xc, yc, w, h = parse.parse(cls.format_str, bbox_str)
        return cls(xc, yc, w, h)


class DetObject:
    format_str = "name={}, score={:0.3f}, bbox={}"

    def __init__(self, bbox: DetBbox, score: float, name: str):
        self.bbox = bbox
        self.score = score
        self.name = name

    def __str__(self) -> str:
        return self.format_str.format(self.name, self.score, self.bbox)

    @classmethod
    def from_string(cls, obj_str):
        name, score, bbox_str = parse.parse(cls.format_str, obj_str)
        bbox = DetBbox.from_string(bbox_str)
        obj = cls(bbox, score, name)
        return obj
