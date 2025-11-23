B_GIRDER = "girder"
B_PILE_LEFT = "pile_left"
B_PILE_RIGHT = "pile_right"
B_STRUT_LEFT = "strut_left"
B_STRUT_RIGHT = "strut_right"
B_STRUT_INNER = "strut_inner"


P_GIRDER_LEFT = "girder_left"
P_GIRDER_STRUT_LEFT = "girder_strut_left"
P_GIRDER_PILE_LEFT = "girder_pile_left"
P_GIRDER_FIELD_CENTER = "girder_field_center"
P_GIRDER_INNER_STRUT = "girder_inner_strut"
P_GIRDER_PILE_RIGHT = "girder_pile_right"
P_GIRDER_STRUT_RIGHT = "girder_strut_right"
P_GIRDER_RIGHT = "girder_right"
# helper points for wind loads

POINTS_GIRDER = [
    P_GIRDER_LEFT,
    P_GIRDER_FIELD_CENTER,
    P_GIRDER_STRUT_LEFT,
    P_GIRDER_PILE_LEFT,
    P_GIRDER_INNER_STRUT,
    P_GIRDER_PILE_RIGHT,
    P_GIRDER_STRUT_RIGHT,
    P_GIRDER_RIGHT,
]


P_PILE_LEFT_BOTTOM = "pile_left_bottom"
# connection point of the outer or inner strut
P_PILE_LEFT_STRUT = "pile_left_strut"
P_PILE_LEFT_TOP = "pile_left_top"

POINTS_PILE_LEFT = [P_PILE_LEFT_BOTTOM, P_PILE_LEFT_STRUT, P_PILE_LEFT_TOP]


P_PILE_RIGHT_BOTTOM = "pile_right_bottom"
# connection point of the outer or inner strut
P_PILE_RIGHT_STRUT = "pile_right_strut"
P_PILE_RIGHT_TOP = "pile_right_top"

POINTS_PILE_RIGHT = [P_PILE_RIGHT_BOTTOM, P_PILE_RIGHT_STRUT, P_PILE_RIGHT_TOP]

P_LEFT_SUPPORT = P_PILE_LEFT_BOTTOM
P_RIGHT_SUPPORT = P_PILE_RIGHT_BOTTOM
