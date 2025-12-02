# JSON API

* Units:
  * kg
  * kN
  * m
  * ° (angle)

GET `http://<server>/api` Content-Type: `application/json`

*Parameters:*
```json
{
  "system": "TWOPILE_WITH_TWO_STRUTS",
  "beam_parameters": {
    "girder": {
      "material": {
        "E": <young_modulus_kn_m2>,
        "G": <shear_modulus_kn_m2>,
        "rho": <mass_density_kg_m3>
      },
      "cross_section": {
        "A": <cross_section_area_m2>,
        "Iy": <I-value_y-axis_m4>,
        "Iz": <I-value_z-axis_m4>,
      }
    },
    "pile_left": { ... },   // or specify each pile separately
    "pile_right": { ... },
    "strut_left": { ... },  // or specify each strut separately
    "strut_right": { ... }
  },
  "geometry": {
    "rigid_supports": <bool>,
    "alpha": <number:ground_slope_angle>,
    "beta": <number:solar_panels_inclination>,
    "t": <number>,    // 'Nicht nutzbare Bodenschicht'
    "h": <number>,    // height of left (long) pile
    "o_l": <number>,  // overhang left
    "o_r": <number>,  // overhang right
    "a": <number>,    // distance between piles
    "s_l": <number>,  // distance connection point of left strut from top of pile
    "s_r": <number>,  // distance connection point of right strut from top of pile
  },
  "load": {
    "e": <number>,       // pile distance in direction of purlins
    "Gk": <number>,      // self wieght solar modules kN/m^2
    "Sk": <number>,      // self weight snow kN/m^2
    "Wdown": <number>,   // wind pressure kN/m^2 ("simple" variant)
    "Wdown": [
      { "fraction": 0.4, "load": 0.8 },
      { "fraction" : 0.3, "load": 1.3 },
      { "fraction": 0.3, "load": 1.5 }
    ], // wind pressure in 3 sections
    // top 40% of the table have 0.8 kN/m2
    // middle 30% of the table have 1.3 kN/m2
    // bottom 30% of the table have 1.5 kN/m2
    "Wup": -1.0 //wind suction (simple variant). Section wise definition analogous to wind pressure (but with negative load)
  },
  "calculation": {
    "nelem": <number: number of FEM nodes, default: 40>,
    "gzg": <bool: include gzg calculation, default: True>,
    "gzt": <bool: include gzt calculation, default: True>,
  }
}
```

Response:
- OK case:
  
  `HTTP 200`
  ```json
  {
    "results": {
      "gzt": {
        "internal_forces": {
          "beams": {
            "girder": {
              "BEAM_MAXIMA": {
                "N": [-10, 10], // Array with 1 or 2 values: maximal negative (if exists), maximal positive (if exists)
                "Vz": [-8, 16.2],
                "My": [7.3] // No maximal negative value, so only 1 value
              },
              "girder_left": { // first point on girder, see constants.py for point names
                "N": [-3.5], // No maximal positive value, so only one value
                "Vz": [-3.2, 33.2],
                "My": [-1, 2.3]
              },
              "girder_strut_left": {
                ...
              },
              ...
            },
            "pile_left": {
              ...
            },
            ...
          }
        },
        "support_forces": {
          "pile_left_bottom": {
            "Fx": [-1.30, 2.3],
            "Fz": [20.3],
            "My": [-2.8]
          },
          "pile_right_bottom": { ... }
        }
      },
      "gzg": {
        "displacements": {
          "beams": {
            "girder": {
              "BEAM_MAXIMA": { 
                "global": {
                  "Ux": <number>,
                  "Uz": <number>
                },
                "local": {
                  "ux": <number>,
                  "uz": <number>
                }
              },
              "girder_left": { // first point on girder
                "global": {
                  "Ux": <number>,
                  "Uz": <number>
                },
                "local": { ...}
              },
              "girder_strut_left": { ... } //second point on girder and so on....
            },
            "pile_left": { ... },
            ...
          }
        }
      }
    }
  }
  ```
- Error case:
  
  `HTTP 500` (server error)