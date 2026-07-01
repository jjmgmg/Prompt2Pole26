# Prompt2Pole — agente de conducción para TORCS (SCR)

Agente de aprendizaje por refuerzo (SAC, Stable-Baselines3) que conduce el coche
`car1-ow1` en el circuito **Corkscrew** de **TORCS** a través del interfaz **SCR**,
usando únicamente **sensores no visuales**. Se conecta como cliente al `scr_server`
y decide dirección y acelerador/freno en cada tic; la marcha la gestiona una caja
de cambios automática por velocidad.

## Requisitos

- **Python 3.12**
- **TORCS** con el `scr_server` (interfaz SCR), coche `car1-ow1`.
- Dependencias de Python:
  ```bash
  pip install -r requirements.txt
  ```
  (inferencia en **CPU**; no requiere GPU.)

## Cómo ejecutar

1. Lanza TORCS con una carrera en **Corkscrew** con `car1-ow1`, usando el
   `scr_server` a la escucha en el puerto **3001**.
2. Arranca el agente:
   ```bash
   python torcs_jm_par.py            # puerto 3001 por defecto
   python torcs_jm_par.py -p 3001    # puerto explícito
   ```

El checkpoint se carga de `models/sac_975000.zip`; se puede sobreescribir con la
variable de entorno `P2P_MODEL`. Para un volcado de diagnóstico por decisión,
`P2P_DEBUG=1`.

## Contenido

| Ruta | Qué es |
|---|---|
| `torcs_jm_par.py` | Punto de entrada: transporte SCR + bucle que conduce con el agente. |
| `p2p_agent.py` | Agente: observación → política → acción (dirección, acelerador/freno, marcha). |
| `models/sac_975000.zip` | Pesos del agente entrenado (SAC). |
| `prompt2pole/` | Módulos mínimos de inferencia (observación, espacios, caja de cambios, política). |
| `car1-ow1_prompt2pole26.rgb` | Livery del coche. |
| `requirements.txt` | Dependencias de Python. |

## Más información

Vídeos, contacto y material adicional: <https://linktr.ee/prompt2pole>

## Licencia

Ver [LICENSE](LICENSE).
