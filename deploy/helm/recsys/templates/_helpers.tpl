{{/* Chart name. */}}
{{- define "recsys.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Fully qualified app name (release-prefixed, 63-char DNS limit). */}}
{{- define "recsys.fullname" -}}
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

{{- define "recsys.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Common labels. */}}
{{- define "recsys.labels" -}}
helm.sh/chart: {{ include "recsys.chart" . }}
app.kubernetes.io/name: {{ include "recsys.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: mlops-recommendation-pipeline
{{- end }}

{{/* Selector labels for a component (api | consumer). */}}
{{- define "recsys.selectorLabels" -}}
app.kubernetes.io/name: {{ include "recsys.name" .root }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{- define "recsys.image" -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) }}
{{- end }}

{{/* Environment shared by all containers. */}}
{{- define "recsys.envFrom" -}}
- configMapRef:
    name: {{ include "recsys.fullname" . }}
{{- if .Values.existingSecret }}
- secretRef:
    name: {{ .Values.existingSecret }}
    optional: true
{{- end }}
{{- end }}
