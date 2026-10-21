FROM ultralytics/ultralytics:latest-nvidia-arm64

USER root

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN python -m pip install --no-cache-dir ipykernel

# Reuse an existing group/user when the image already owns these IDs.
RUN set -eux; \
    old_group="$(getent group "$USER_GID" | cut -d: -f1)"; \
    if [ -z "$old_group" ]; then \
        groupadd --gid "$USER_GID" "$USERNAME"; \
    elif [ "$old_group" != "$USERNAME" ]; then \
        groupmod --new-name "$USERNAME" "$old_group"; \
    fi; \
    old_user="$(getent passwd "$USER_UID" | cut -d: -f1)"; \
    if [ -z "$old_user" ]; then \
        useradd --create-home --uid "$USER_UID" \
            --gid "$USER_GID" --shell /bin/bash "$USERNAME"; \
    else \
        if [ "$old_user" != "$USERNAME" ]; then \
            usermod --login "$USERNAME" "$old_user"; \
        fi; \
        usermod --gid "$USER_GID" --shell /bin/bash \
            --home "/home/$USERNAME" --move-home "$USERNAME"; \
    fi; \
    mkdir -p "/home/$USERNAME"; \
    chown -R "$USER_UID:$USER_GID" "/home/$USERNAME"

ENV HOME=/home/${USERNAME}
ENV YOLO_CONFIG_DIR=/home/${USERNAME}/.config/Ultralytics

USER ${USERNAME}
WORKDIR /workspace/cv-coco-detection