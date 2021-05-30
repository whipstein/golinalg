package goblas

import (
	"bytes"
)

type memory struct {
	sslct struct {
		selopt int
		seldim int
		selval []bool
		selwr  []float32
		selwi  []float32
	}
	cenvir struct {
		nproc  int
		nshift int
		maxb   int
	}
	claenv struct {
		iparms []int
	}
	combla struct {
		icase int
		_case string
		n     int
		incx  int
		incy  int
		off   int
		pass  bool
		nout  *writer
	}
	begc struct {
		mi int
		mj int
		ic int
		i  int
		j  int
	}
	infoc struct {
		infot int
		nout  *writer
		nunit int
		ok    bool
		lerr  bool
		noutc int
	}
	srnamc struct {
		srnamt []byte
	}
}

var common memory

var inegtwo, inegone, izero, ione, itwo, ithree, ifour, ifive, isix, iseven, ieight, inine, iten int = -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
var snegone, szero, sone, stwo, sthree, sfour, sfive, ssix, sseven, seight, snine, sten float32 = -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

func transposeByte(a *[][]byte) [][]byte {
	b := make([][]byte, len((*a)[0]))
	for i := range b {
		b[i] = make([]byte, len(*a))
		for j := range b[i] {
			b[i][j] = (*a)[j][i]
		}
	}
	// copy(*a, b)
	return b
}

func transposeFloat32(a *[][]float32) [][]float32 {
	b := make([][]float32, len((*a)[0]))
	for i := range b {
		b[i] = make([]float32, len(*a))
		for j := range b[i] {
			b[i][j] = (*a)[j][i]
		}
	}
	// copy(*a, b)
	return b
}

// func transposeFloat32x(a *[][][]float32) [][][]float32 {
// 	b := make([][][]float32, len((*a)[0][0]))
// 	for i := range b {
// 		b[i] = make([][]float32, len((*a)[0]))
// 		for j := range b[i] {
// 			b[i][j] = make([]float32, len(*a))
// 			for k := range b[i][j] {
// 				b[i][j][k] = (*a)[k][j][i]
// 			}
// 		}
// 	}
// 	// copy(*a, b)
// 	return b
// }

func lentrim(a *[]byte) int {
	return len(bytes.TrimSpace(*a))
}

func maxlocf32(a []float32) (idx int) {
	if len(a) == 1 {
		return
	}
	amax := a[0]
	for i := 1; i < len(a); i++ {
		if a[i] > amax {
			idx = i
			amax = a[i]
		}
	}
	return
}

// func sgenSubslice1D(in *[]float32, idx interface{}) *[]float32 {
// 	var i int

// 	ix := reflect.ValueOf(idx)

// 	switch idx.(type) {
// 	case *int:
// 		x := reflect.Indirect(ix)
// 		i = int(x.Int())
// 	case int:
// 		i = int(ix.Int())
// 	}

// 	var out []float32
// 	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&out))
// 	hdr.Len = len(*in) - i
// 	hdr.Cap = cap(*in) - i
// 	hdr.Data = uintptr(unsafe.Pointer(&(*in)[i]))
// 	return &out
// }

// func sgenSubslice2Dto1D(in *[][]float32, row, col interface{}) *[]float32 {
// 	var r, c int

// 	rx := reflect.ValueOf(row)
// 	cx := reflect.ValueOf(col)

// 	switch row.(type) {
// 	case *int:
// 		x := reflect.Indirect(rx)
// 		r = int(x.Int())
// 	case int:
// 		r = int(rx.Int())
// 	}

// 	switch col.(type) {
// 	case *int:
// 		x := reflect.Indirect(cx)
// 		c = int(x.Int())
// 	case int:
// 		c = int(cx.Int())
// 	}

// 	var out []float32
// 	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&out))
// 	hdr.Len = len((*in))
// 	hdr.Cap = cap((*in))
// 	hdr.Data = uintptr(unsafe.Pointer(&(*in)[r][c]))
// 	return &out
// }

// func sgenSubslice2D(in *[][]float32, row, col interface{}) *[][]float32 {
// 	var r, c, rows, cols int

// 	rx := reflect.ValueOf(row)
// 	cx := reflect.ValueOf(col)

// 	switch row.(type) {
// 	case *int:
// 		x := reflect.Indirect(rx)
// 		r = int(x.Int())
// 	case int:
// 		r = int(rx.Int())
// 	}

// 	switch col.(type) {
// 	case *int:
// 		x := reflect.Indirect(cx)
// 		c = int(x.Int())
// 	case int:
// 		c = int(cx.Int())
// 	}

// 	rows = len(*in) - r
// 	cols = len((*in)[0]) - c

// 	var out [][]float32 = make([][]float32, rows)
// 	for j := 0; j < rows; j++ {
// 		hdr := (*reflect.SliceHeader)(unsafe.Pointer(&out[j]))
// 		hdr.Len = cols
// 		hdr.Cap = cols
// 		hdr.Data = uintptr(unsafe.Pointer(&(*in)[r][c]))
// 	}

// 	return &out
// }

// func sgen1Dto2D(in *[]float32, offset, ldin interface{}) *[][]float32 {
// 	var off, ld, rows int

// 	offsetx := reflect.ValueOf(offset)
// 	ldx := reflect.ValueOf(ldin)

// 	switch offset.(type) {
// 	case *int:
// 		x := reflect.Indirect(offsetx)
// 		off = int(x.Int())
// 	case int:
// 		off = int(offsetx.Int())
// 	}

// 	switch ldin.(type) {
// 	case *int:
// 		x := reflect.Indirect(ldx)
// 		ld = int(x.Int())
// 	case int:
// 		ld = int(ldx.Int())
// 	}

// 	elemlen := uintptr(unsafe.Pointer(&(*in)[1])) - uintptr(unsafe.Pointer(&(*in)[0]))
// 	rows = int((len(*in) - off) / ld)

// 	var out [][]float32 = make([][]float32, rows)
// 	for j := 0; j < rows; j++ {
// 		hdr := (*reflect.SliceHeader)(unsafe.Pointer(&out[j]))
// 		hdr.Len = ld
// 		hdr.Cap = ld
// 		hdr.Data = uintptr(unsafe.Pointer(&(*in)[off])) + uintptr(uint(j*ld)*uint(elemlen))
// 	}

// 	return &out
// }

// func sgen2Dto1D(in *[][]float32, ldin interface{}) *[]float32 {
// 	var ld int

// 	ldx := reflect.ValueOf(ldin)

// 	switch ldin.(type) {
// 	case *int:
// 		x := reflect.Indirect(ldx)
// 		ld = int(x.Int())
// 	case int:
// 		ld = int(ldx.Int())
// 	}

// 	out := func() *[]float32 {
// 		arr := make([]float32, len((*in)[0])+len(*in)*ld)
// 		for j := 0; j < len(*in); j++ {
// 			for i := 0; i < len((*in)[0]); i++ {
// 				arr[i+j*ld] = (*in)[j][i]
// 			}
// 		}
// 		return &arr
// 	}()

// 	return out
// }
