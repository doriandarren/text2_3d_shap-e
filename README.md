# Project

proyecto: hysts/Shap-E

Puerto: 7861


## Instalación

```sh
crear carpeta en la raíz: "outputs"
```



```sh
netsh advfirewall firewall add rule name="ShapE_API_7861" dir=in action=allow protocol=TCP localport=7861



docker compose down
docker compose up -d --build

docker compose up --build



docker compose up
docker compose down



curl -X POST "http://localhost:7861/generate" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"a cute low-poly fox sitting on a rock\", \"seed\": 0}" \
  --output fox.glb



curl -OJ -X POST "http://localhost:7861/generate" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"low-poly viking warrior standing proudly, wearing a horned helmet and fur cloak, holding a round wooden shield and a battle axe, strong muscular build, long braided beard, Norse warrior aesthetic, fantasy style, slightly stylized proportions, clean geometry and flat colors, game-ready\", \"seed\": 42}" \
  --output viking.glb




Invoke-WebRequest `
  -Uri "http://localhost:7861/generate" `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"prompt":"a cute low-poly fox sitting on a rock","seed":0}' `
  -OutFile "C:\ai\text2_3d_shap-e\outputs\fox.glb"



```
