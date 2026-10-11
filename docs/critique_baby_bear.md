# Baby‑bear critique (design review, 2026‑09‑24)

An expert organic‑robot design critique, and what was done about each point.

## First impression before the change
It read as a *"raptor skull on a lab rig"*:
- a small faceted head held low and forward, with a long toothed snout;
- a single camera "third eye";
- skeletal lattice legs;
- walking‑stick feet;
- an exposed Jetson;
- random colours;
- pointed cut‑outs everywhere.

## Baby‑schema principles applied
- **Head size:** a big head, at least as wide as the body.
- **Face:** a big forehead, eyes at or below mid‑head, a short muzzle.
- **Shapes:** convex everywhere, with no pointed or concave silhouettes.
- **Limbs:** chunky limbs, oversized paws, a low compact body with a stub tail.
- **Colour:** one warm colour, black eyes and nose, matte.
- **Posture:** head up and slightly tilted, mouth closed.

## Changes made
| critique item | done |
|---|---|
| Stance: crouch to about 145 mm, head up about 12° | stand pose (0, −49.3°, 92.8°), neck −12° |
| New head: about 95 wide, flat crown, 45° shoulders, round cheeks | `head`: 95 × 76 × 105 mm; shoulders graded 45→53→63→76° so the top reads round but still prints crown‑down |
| Camera as one eye, matching second eye | left eye = lens; Ø22 sockets on both sides |
| Short muzzle (about 20 mm), nose on the upper front edge | 16 mm muzzle with a 55° top, rounded‑triangle nose |
| Jaw as a round chin pod; delete the sawtooth; canines hidden when closed | `jaw`: round pod; 2 + 2 × 4 mm canines that sit in pockets when closed |
| Round ears on a crown plate | `ear_cap`: two Ø26 cupped ears, splayed 15° |
| Back hump over the Jetson, intake gap, exhaust vents | `hump` part; 3 mm rim gap, 21 round vents |
| Honey/caramel matte, brown paws, black eyes/nose | render colours and README |
| Bean paws with toe bumps | `foot_left` / `foot_right`: Ø42 TPU, three toes facing forward |
| Loaf body section, solid walls, round vents | loaf profile (belly r 13), Ø6 vents, round skin holes |
| Stub tail on the rear face | Ø16 gumdrop, printed with the body |
| Chubbier legs: fill the thigh lattice, knee domes, fat shins | closed thigh skin, 45° cone "buttons" on the hip‑pitch and knee idler horns, ellipsoid shin about 24 mm wide |
| No pointed cut‑outs on the outside | sleeve windows removed; visible holes round |

## Not done / trade‑offs
- The DS3218s are too weak for fast snappy motion anyway. Keep head and jaw moves eased, add an idle "breathing" sway, and glance away now and then (software).
- A white catch‑light dot is left to paint.
