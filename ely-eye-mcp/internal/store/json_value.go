package store

import (
	"encoding/json"
	"errors"
)

func optionalJSONValue(value string) (any, error) {
	if value == "" || value == "null" {
		return nil, nil
	}
	return decodeJSONValue(value)
}

func requiredJSONValue(value string) (any, error) {
	if value == "" {
		return map[string]any{}, nil
	}
	return decodeJSONValue(value)
}

func readJSONValue(data []byte) (any, error) {
	if !json.Valid(data) {
		return nil, errors.New("invalid JSON")
	}
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		return nil, err
	}
	return value, nil
}

func decodeJSONValue(value string) (any, error) {
	return readJSONValue([]byte(value))
}
