M_PER_CM = 0.01
M2_PER_CM2 = M_PER_CM**2
M3_PER_CM3 = M_PER_CM**3
M4_PER_CM4 = M_PER_CM**4
MM2_PER_M2 = 1_000_000


class Vector:
    """Represents a cartesian 3 dimensional vector"""

    def __init__(self, c_list):
        self.c_list = [float(e) for e in c_list]

    @classmethod
    def xz(cls, x=0.0, z=0.0):
        return cls([x, 0.0, z])

    @property
    def coords(self):
        return self.c_list

    @property
    def x(self):
        return self.c_list[0]

    @property
    def y(self):
        return self.c_list[1]

    @property
    def z(self):
        return self.c_list[2]

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector([l[0] + l[1] for l in zip(self.coords, other.coords)])
        if isinstance(other, list) and len(other) == 3:
            return Vector([l[0] + l[1] for l in zip(self.coords, other)])
        return NotImplemented

    def __sub__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented

        return Vector([l[0] - l[1] for l in zip(self.coords, other.coords)])

    def __mul__(self, other):
        return Vector([other * e for e in self.coords])

    __rmul__ = __mul__


class ParametricVector:
    """A three dimensional vector parameterized by one or more parameters"""

    def __init__(self, lambda_x, lambda_y, lambda_z):
        self.x = lambda_x
        self.y = lambda_y
        self.z = lambda_z

    def get(self, *args, **kwargs) -> Vector:
        return Vector(
            [
                self.x(*args, **kwargs) if self.x else 0.0,
                self.y(*args, **kwargs) if self.y else 0.0,
                self.z(*args, **kwargs) if self.z else 0.0,
            ]
        )

    @classmethod
    def xz(cls, lambda_x, lambda_z):
        return cls(lambda_x, None, lambda_z)

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)
