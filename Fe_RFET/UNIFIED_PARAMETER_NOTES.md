# FeRFET unified N/P parameter model

## Scope

This variant removes the separate N-type and P-type parameter sets. The
ferroelectric polarization sign still selects the electrical polarity and
current sign, but both polarities use the same compact-model parameter values.

The original `Fe_RFET_tcad_optimized.zip` is not modified.

## Current path

```text
FE polarization -> electrostatic S/D doping
BSIM-CMG drift-diffusion channel
Schottky-contact tunneling series limit
```

The optional direct source-to-drain tunneling model and all of its parameters,
variables, operating-point outputs, equations, and terminal-current
contributions were removed. Schottky-contact tunneling remains enabled by
default with `rfettunmod=1`.

The terminal conduction current is

```text
Icond = IDD * Itun / (IDD + Itun + Ifloor)
```

## Defaults from paper45

The following common N/P defaults come from the table captioned
"Extracted model parameter samples of the FeRFET device":

| Parameter | Unified default |
|---|---:|
| `phig` | 4.5 eV |
| `cit` | 2.62e-4 F/m^2 |
| `cdsc` | 1.84 F/m^2 |
| `vsat` | 7.5e5 m/s |
| `dvt0` | 3.0 |
| `dsub` | 0.478 |
| `eta0` | 0.0766 |
| `rfetTunAmp` | 1.02e4 A/m |
| `rfetTunSlope` | 2.85e9 V/m |
| `rfetBarrier` | 0.562 eV |

## Defaults from the BSIM-CMG 112.0.0 Technical Manual

Parameters absent from the paper table use the manual defaults:

| Parameter | Unified default | Manual page |
|---|---:|---:|
| `cdscd` | 7e-3 F/m^2 | 129 |
| `ksativ` | 1.0 | 131 |
| `u0` | 3e-2 m^2/(V*s) | 131 |
| `rsw` | 50 ohm*um^wr | 133 |
| `rdw` | 50 ohm*um^wr | 133 |

## RFET-specific common defaults

The BSIM-CMG manual does not define the remaining RFET-contact parameters.
Pairs that already had identical N/P values were collapsed to that value:

| Parameter | Unified default |
|---|---:|
| `rfetTunBarrierPower` | 1.5 |
| `rfetTunLength` | 5e-9 m |
| `rfetContactNFactor` | 1.0 |
| `rfetFieldScale` | 1.0 |
| `rfetGateFieldFactor` | 0.0 |
| `rfetDrainCoupling` | 0.0 |
| `rfetFieldOffset` | 0.0 V |
| `rfetTunVoffset` | 0.0 V |

`rfetContactWf=4.63478 eV` is the common midgap workfunction derived from the
model material values `easub=4.0727 eV` and `bg0sub=1.12416 eV`. With
`rfetBarrierMode=1`, it produces equal electron and hole barrier magnitudes.
The default `rfetBarrierMode=0` uses the explicit paper value
`rfetBarrier=0.562 eV` directly.

