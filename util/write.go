package util

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

type Reader struct {
	f   *os.File
	buf *bufio.Reader
	eof bool
}

func NewReader(filename string) *Reader {
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

	r := Reader{f: f, buf: buf, eof: false}
	return &r
}

func (r *Reader) Readln(format string, a ...interface{}) []byte {
	line, _, err := r.buf.ReadLine()
	if err != nil && err != io.EOF {
		log.Fatal(err)
	} else if err == io.EOF {
		r.eof = true
		return line
	} else if len(a) < 1 {
		return line
	}

	_, err = fmt.Sscanf(string(line), format, a...)
	if err != nil {
		if err == io.EOF {
			r.eof = true
			return line
		}
		log.Fatal(err)
	}
	return line
}

func (r *Reader) Readlnraw(line *[]byte) {
	var err error
	*line, _, err = r.buf.ReadLine()
	if err != nil && err != io.EOF {
		log.Fatal(err)
	} else if err == io.EOF {
		r.eof = true
	}
}

func (r *Reader) Readval(format string, a ...interface{}) {
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

type Writer struct {
	f   *os.File
	buf *bufio.Writer
}

func NewWriter(filename string) *Writer {
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

	w := Writer{f: f, buf: buf}
	return &w
}

func (w *Writer) Write(format string, a ...interface{}) {
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
