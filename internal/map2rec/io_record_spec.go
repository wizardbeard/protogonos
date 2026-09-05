package map2rec

import (
	"fmt"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func SensorIORecordSpec(record SensorRecord) model.IORecordSpec {
	return model.IORecordSpec{
		Name:          protoio.CanonicalSensorName(record.Name),
		ReferenceName: record.Name,
		Type:          record.Type,
		ScapeKind:     scapeKind(record.Scape),
		ScapeName:     scapeName(record.Scape),
		Format:        stringField(record.Format),
		VL:            record.VL,
		Parameters:    stringParameters(record.Parameters),
	}
}

func ActuatorIORecordSpec(record ActuatorRecord) model.IORecordSpec {
	return model.IORecordSpec{
		Name:          protoio.CanonicalActuatorName(record.Name),
		ReferenceName: record.Name,
		Type:          record.Type,
		ScapeKind:     scapeKind(record.Scape),
		ScapeName:     scapeName(record.Scape),
		Format:        stringField(record.Format),
		VL:            record.VL,
		Parameters:    stringParameters(record.Parameters),
	}
}

func scapeKind(value any) string {
	items, ok := asAnySlice(value)
	if !ok || len(items) == 0 {
		return stringField(value)
	}
	return stringField(items[0])
}

func scapeName(value any) string {
	items, ok := asAnySlice(value)
	if !ok || len(items) < 2 {
		return ""
	}
	return stringField(items[1])
}

func stringField(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		return fmt.Sprint(v)
	}
}

func stringParameters(value any) []string {
	items, ok := asAnySlice(value)
	if !ok {
		if value == nil {
			return nil
		}
		return []string{stringField(value)}
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		out = append(out, stringField(item))
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
