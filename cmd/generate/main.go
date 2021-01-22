package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"text/template"

	"go.uber.org/multierr"
)

func run() error {
	structName := ""
	fileName := os.Getenv("GOFILE")
	packageName := os.Getenv("GOPACKAGE")

	flag.StringVar(&structName, "struct", "", "struct to be generated for")
	flag.Parse()

	if structName == "" || fileName == "" || packageName == "" {
		return fmt.Errorf("missing arguments or environment variables")
	}

	log.Printf("go-featureprocessing is writing struct transfomer for struct '%s' $GOFILE=%s $GOPACKAGE=%s ", structName, fileName, packageName)

	inputCode, err := ioutil.ReadFile(fileName)
	if err != nil {
		return fmt.Errorf("can not open input file: %w", err)
	}

	params, err := parseCode(fileName, inputCode, structName, packageName)
	if err != nil {
		return fmt.Errorf("can not parse code: %w", err)
	}

	codeFilePath := fmt.Sprintf("%sfp.go", strings.ToLower(structName))
	testFilePath := fmt.Sprintf("%sfp_test.go", strings.ToLower(structName))

	if err := generate(params, codeFilePath, "templateCode", templateCode); err != nil {
		return fmt.Errorf("can not make code: %w", err)
	}
	if err := generate(params, testFilePath, "templateTests", templateTests); err != nil {
		return fmt.Errorf("can not make tests: %w", err)
	}

	return nil
}

func generate(params *TemplateParams, outfilepath string, templateName string, templateVal string) error {
	code := bytes.NewBufferString("")
	parsedTemplate, err := template.New(templateName).Parse(templateVal)
	if err != nil {
		return fmt.Errorf("can not initialize template: %w", err)
	}
	if err := parsedTemplate.Execute(code, params); err != nil {
		return fmt.Errorf("can not execute template: %w", err)
	}

	if err := writeCodeToFile(code.Bytes(), outfilepath); err != nil {
		return fmt.Errorf("can not write code: %w", err)
	}
	return nil
}

func writeCodeToFile(code []byte, outfilepath string) (err error) {
	formattedCode, err := format.Source(code)
	if err != nil {
		return fmt.Errorf("can not format code: %w, code: %s", err, code)
	}

	if err := os.MkdirAll(filepath.Dir(outfilepath), 0700); err != nil {
		return fmt.Errorf("can not make dir for output file: %w", err)
	}

	file, err := os.Create(outfilepath)
	if err != nil {
		return fmt.Errorf("can not create file: %w", err)
	}
	defer func() { err = multierr.Combine(err, file.Close()) }()

	if _, err := file.Write(formattedCode); err != nil {
		return fmt.Errorf("can not write code to file: %w", err)
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		log.Fatalf(fmt.Errorf("go-featureprocessing encountered error: %w", err).Error())
	}
}
