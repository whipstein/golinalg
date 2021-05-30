package goblas

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

type reader struct {
	f   *os.File
	buf *bufio.Reader
	eof bool
}

func newReader(filename string) *reader {
	var f *os.File
	var err error

	if filename != "" {
		f, err = os.Open(filename)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		f = os.Stdin
	}

	buf := bufio.NewReader(f)

	r := reader{f: f, buf: buf, eof: false}
	return &r
}

func (r *reader) readln(format string, a ...interface{}) {
	line, _, err := r.buf.ReadLine()
	if err != nil && err != io.EOF {
		log.Fatal(err)
	} else if err == io.EOF {
		r.eof = true
		return
	}

	_, err = fmt.Sscanf(string(line), format, a...)
	if err != nil {
		if err == io.EOF {
			r.eof = true
			return
		}
		log.Fatal(err)
	}
}

func (r *reader) readlnraw(line *[]byte) {
	var err error
	*line, _, err = r.buf.ReadLine()
	if err != nil && err != io.EOF {
		log.Fatal(err)
	} else if err == io.EOF {
		r.eof = true
	}
}

func (r *reader) readval(format string, a ...interface{}) {
	var line string
	var err error

	for {
		line, err = r.buf.ReadString(' ')
		if err != nil && err != io.EOF {
			log.Fatal(err)
		} else if err == io.EOF {
			r.eof = true
			return
		} else if strings.TrimSpace(line) != "" {
			break
		}
	}

	_, err = fmt.Sscanf(line, format, a...)
	if err != nil {
		log.Fatal(err)
	}
}

type writer struct {
	f   *os.File
	buf *bufio.Writer
}

func newWriter(filename string) *writer {
	var f *os.File
	var err error

	if filename != "" {
		f, err = os.Create(filename)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		f = os.Stdout
	}

	buf := bufio.NewWriter(f)

	w := writer{f: f, buf: buf}
	return &w
}

func (w *writer) write(format string, a ...interface{}) {
	line := fmt.Sprintf(format, a...)
	_, err := w.buf.WriteString(line)
	if err != nil {
		log.Fatal(err)
	}
	err = w.buf.Flush()
	if err != nil {
		log.Fatal(err)
	}
}
