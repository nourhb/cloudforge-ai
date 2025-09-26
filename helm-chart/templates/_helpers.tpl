{{- define "cloudforge-ai.name" -}}
cloudforge-ai
{{- end -}}

{{- define "cloudforge-ai.fullname" -}}
{{- printf "%s" (include "cloudforge-ai.name" . ) -}}
{{- end -}}

# TEST: Helpers for Helm templating
