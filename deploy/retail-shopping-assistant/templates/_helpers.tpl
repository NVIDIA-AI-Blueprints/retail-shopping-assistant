{{/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "retail-shopping-assistant.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "retail-shopping-assistant.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "retail-shopping-assistant.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "retail-shopping-assistant.labels" -}}
helm.sh/chart: {{ include "retail-shopping-assistant.chart" . }}
{{ include "retail-shopping-assistant.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: retail-shopping-assistant
{{- end }}

{{/*
Selector labels
*/}}
{{- define "retail-shopping-assistant.selectorLabels" -}}
app.kubernetes.io/name: {{ include "retail-shopping-assistant.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "retail-shopping-assistant.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "retail-shopping-assistant.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the namespace
*/}}
{{- define "retail-shopping-assistant.namespace" -}}
{{- if .Values.global.namespace }}
{{- .Values.global.namespace }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}

{{/*
Return the storage class name
*/}}
{{- define "retail-shopping-assistant.storageClass" -}}
{{- if .Values.global.storageClass }}
{{- .Values.global.storageClass }}
{{- end }}
{{- end }}

{{/*
Return the RWX storage class name
*/}}
{{- define "retail-shopping-assistant.storageClassRWX" -}}
{{- if .Values.global.storageClassRWX }}
{{- .Values.global.storageClassRWX }}
{{- end }}
{{- end }}

{{/*
Return image pull secrets
*/}}
{{- define "retail-shopping-assistant.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.global.imagePullSecrets }}
  - name: {{ .name }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chain Server service name
*/}}
{{- define "retail-shopping-assistant.chainServer.name" -}}
{{- printf "%s-chain-server" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Catalog Retriever service name
*/}}
{{- define "retail-shopping-assistant.catalogRetriever.name" -}}
{{- printf "%s-catalog-retriever" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Memory Retriever service name
*/}}
{{- define "retail-shopping-assistant.memoryRetriever.name" -}}
{{- printf "%s-memory-retriever" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Rails service name
*/}}
{{- define "retail-shopping-assistant.rails.name" -}}
{{- printf "%s-rails" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Frontend service name
*/}}
{{- define "retail-shopping-assistant.frontend.name" -}}
{{- printf "%s-frontend" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Nginx service name
*/}}
{{- define "retail-shopping-assistant.nginx.name" -}}
{{- printf "%s-nginx" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
etcd service name
*/}}
{{- define "retail-shopping-assistant.etcd.name" -}}
{{- printf "%s-etcd" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
MinIO service name
*/}}
{{- define "retail-shopping-assistant.minio.name" -}}
{{- printf "%s-minio" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Milvus service name
*/}}
{{- define "retail-shopping-assistant.milvus.name" -}}
{{- printf "%s-milvus" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
PostgreSQL service name
*/}}
{{- define "retail-shopping-assistant.postgresql.name" -}}
{{- printf "%s-postgresql" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Secrets name
*/}}
{{- define "retail-shopping-assistant.secrets.name" -}}
{{- printf "%s-secrets" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
NVIDIA API Keys secret name
*/}}
{{- define "retail-shopping-assistant.nvidiaApiKeys.name" -}}
{{- printf "%s-nvidia-api-keys" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
MinIO credentials secret name
*/}}
{{- define "retail-shopping-assistant.minioCredentials.name" -}}
{{- printf "%s-minio-credentials" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
PostgreSQL credentials secret name
*/}}
{{- define "retail-shopping-assistant.postgresCredentials.name" -}}
{{- printf "%s-postgres-credentials" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Shared data PVC name
*/}}
{{- define "retail-shopping-assistant.sharedDataPvc.name" -}}
{{- printf "%s-shared-data" (include "retail-shopping-assistant.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Component labels
*/}}
{{- define "retail-shopping-assistant.componentLabels" -}}
app: retail-shopping
{{- end }}

{{/*
Pod security context
*/}}
{{- define "retail-shopping-assistant.podSecurityContext" -}}
{{- if .Values.securityContext }}
securityContext:
  {{- toYaml .Values.securityContext | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Container security context
*/}}
{{- define "retail-shopping-assistant.containerSecurityContext" -}}
{{- if .Values.containerSecurityContext }}
securityContext:
  {{- toYaml .Values.containerSecurityContext | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Pod affinity for shared volume (required for RWO storage)
Forces pods to the same node when using ReadWriteOnce volumes
*/}}
{{- define "retail-shopping-assistant.sharedVolumeAffinity" -}}
{{- if and .Values.persistence.sharedData.enabled (eq .Values.persistence.sharedData.accessMode "ReadWriteOnce") }}
affinity:
  podAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app: retail-shopping
        topologyKey: kubernetes.io/hostname
{{- end }}
{{- end }}
