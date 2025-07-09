graph TD;
    A[Inicio] --> B{Actualizar repositorio local};
    B -->|git fetch origin| C{Restablecer al commit deseado};
    C -->|git reset --hard ba80cfeed5c9657112c244e5ae130d963cde4342| D{Limpiar archivos};
    D -->|git clean -df| E{Verificación final};
    E -->|Finalizado| F{🟢 Completado}
