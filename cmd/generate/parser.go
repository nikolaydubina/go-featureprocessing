package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Field represents single transformer and field it transforms, for internal use only
type Field struct {
	Name           string
	Transformer    string
	Expanding      bool
	NumericalInput bool
	TransformerTag string
}

// TemplateParams represents all parameters for template, for internal use only
type TemplateParams struct {
	PackageName              string
	StructName               string
	NumFieldsFlat            int
	Fields                   []Field
	HasLargeTransformers     bool
	HasNumericalTransformers bool
	HasStringTransformers    bool
}

var tagToTransformer = map[string]string{
	"identity":        "Identity",
	"minmax":          "MinMaxScaler",
	"maxabs":          "MaxAbsScaler",
	"standard":        "StandardScaler",
	"quantile":        "QuantileScaler",
	"onehot":          "OneHotEncoder",
	"ordinal":         "OrdinalEncoder",
	"kbins":           "KBinsDiscretizer",
	"countvectorizer": "CountVectorizer",
	"tfidf":           "TFIDFVectorizer",
}

var isTransformerExpanding = map[string]bool{
	"onehot":          true,
	"countvectorizer": true,
	"tfidf":           true,
}

var isTransformerLarge = map[string]bool{
	"quantile":        true,
	"onehot":          true,
	"ordinal":         true,
	"kbins":           true,
	"countvectorizer": true,
	"tfidf":           true,
}

var isTypeSupported = map[string]bool{
	"int":     true,
	"int8":    true,
	"int16":   true,
	"int32":   true,
	"float32": true,
	"float64": true,
	"string":  true,
}

var isTypeNumerical = map[string]bool{
	"int":     true,
	"int8":    true,
	"int16":   true,
	"int32":   true,
	"float32": true,
	"float64": true,
}

// parseCode parses provided at filename or code into AST.
// It finds for struct delcarations matching structName and collects fields information
// that is next used to filling all necessary details for constructing StructTransformer.
func parseCode(filename string, code []byte, structName string, packageName string) (*TemplateParams, error) {
	var err error
	var fields []Field
	numFieldsFlat := 0
	numLargeTransformers := 0
	numNumericalTransformers := 0
	numStringTransformers := 0

	f, err := parser.ParseFile(token.NewFileSet(), filename, code, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("can not parse input file: %w", err)
	}

	ast.Inspect(f, func(node ast.Node) bool {
		decl, ok := node.(*ast.GenDecl)
		if !ok {
			return true
		}

		for _, spec := range decl.Specs {
			typeSpec, ok := spec.(*ast.TypeSpec)
			if !ok {
				continue
			}

			if typeSpec.Name == nil {
				continue
			}

			if typeSpec.Name.Name != structName {
				continue
			}

			structSpec, ok := typeSpec.Type.(*ast.StructType)
			if !ok {
				continue
			}

			for _, field := range structSpec.Fields.List {
				if field == nil {
					continue
				}

				// name
				if len(field.Names) == 0 {
					continue
				}
				name := field.Names[0].Name

				// Field name has to start from UTF-8 letter.
				// This is contraint of Go language spec.
				firstRune, _ := utf8.DecodeRuneInString(name)
				if !unicode.IsLetter(firstRune) {
					continue
				}

				// Should start from latin letter,
				// otherwise some weird error happens with fields inclusion.
				if !unicode.In(firstRune, unicode.Scripts["Latin"]) {
					continue
				}

				// type
				fieldType := field.Type
				if fieldType == nil {
					continue
				}
				fieldTypeIndent := fieldType.(*ast.Ident)
				if fieldTypeIndent == nil {
					continue
				}
				fieldTypeVal := fieldTypeIndent.Name

				// tag
				tagsLit := field.Tag
				if tagsLit == nil {
					continue
				}
				tags := tagsLit.Value

				var tag string
				for _, t := range strings.Fields(strings.Trim(tags, "`")) {
					if strings.HasPrefix(t, "feature:") {
						tag = t
					}
				}
				if tag == "" {
					continue
				}
				tag = strings.Trim(strings.TrimPrefix(tag, "feature:"), "\"")

				if _, ok := tagToTransformer[tag]; !ok {
					err = fmt.Errorf("unexpected value of struct tag \"%s\"", tag)
					return false
				}

				if !isTypeSupported[fieldTypeVal] {
					err = fmt.Errorf("unsupported type %s, supported field types: %#v, note it has to be raw", fieldTypeVal, isTypeSupported)
					return false
				}

				field := Field{
					Name:           name,
					Transformer:    tagToTransformer[tag],
					Expanding:      isTransformerExpanding[tag],
					NumericalInput: isTypeNumerical[fieldTypeVal],
					TransformerTag: tag,
				}
				if !isTransformerExpanding[tag] {
					numFieldsFlat++
				}
				if isTransformerLarge[tag] {
					numLargeTransformers++
				}
				fields = append(fields, field)

				if isTypeNumerical[fieldTypeVal] {
					numNumericalTransformers++
				} else {
					numStringTransformers++
				}
			}

		}
		return true
	})

	params := TemplateParams{
		PackageName:              packageName,
		StructName:               structName,
		NumFieldsFlat:            numFieldsFlat,
		HasLargeTransformers:     numLargeTransformers > 0,
		Fields:                   fields,
		HasNumericalTransformers: numNumericalTransformers > 0,
		HasStringTransformers:    numStringTransformers > 0,
	}

	return &params, err
}
