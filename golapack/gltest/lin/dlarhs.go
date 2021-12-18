package lin

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlarhs chooses a set of NRHS random solution vectors and sets
// up the right hand sides for the linear system
//    op( A ) * X = B,
// where op( A ) may be A or A' (transpose of A).
func Dlarhs(path string, xtype byte, uplo mat.MatUplo, trans mat.MatTrans, m, n, kl, ku, nrhs int, a *mat.Matrix, x *mat.Matrix, b *mat.Matrix, iseed *[]int) (err error) {
	var band, gen, notran, qrs, sym, tran, tri bool
	var diag mat.MatDiag
	var one, zero float64
	var j, mb, nx int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	c1 := path[0]
	c2 := path[1:3]
	tran = trans.IsTrans()
	notran = !tran
	gen = path[1] == 'g'
	qrs = path[1] == 'q' || path[2] == 'q'
	sym = path[1] == 'p' || path[1] == 's'
	tri = path[1] == 't'
	band = path[2] == 'b'
	if c1 != 'D' {
		err = fmt.Errorf("path[0] != 'D': c1='%c'", path[0])
	} else if !(xtype == 'N' || xtype == 'C') {
		err = fmt.Errorf("!(xtype == 'N' || xtype == 'C'): xtype='%c'", xtype)
	} else if (sym || tri) && !(uplo == mat.Upper || uplo == mat.Lower) {
		err = fmt.Errorf("(sym || tri) && !(uplo == mat.Upper || uplo == mat.Lower): path[1]='%c', uplo=%s", path[1], uplo)
	} else if (gen || qrs) && !trans.IsValid() {
		err = fmt.Errorf("(gen || qrs) && !trans.IsValid(): path[1]='%c', trans=%s", path[1], trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if band && kl < 0 {
		err = fmt.Errorf("band && kl < 0: path[2]='%c', kl=%v", path[2], kl)
	} else if band && ku < 0 {
		err = fmt.Errorf("band && ku < 0: path[2]='%c', ku=%v", path[2], ku)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if (!band && a.Rows < max(1, m)) || (band && (sym || tri) && a.Rows < kl+1) || (band && gen && a.Rows < kl+ku+1) {
		err = fmt.Errorf("(!band && a.Rows < max(1, m)) || (band && (sym || tri) && a.Rows < kl+1) || (band && gen && a.Rows < kl+ku+1): path[1]='%c', path[2]='%c', a.Rows=%v, m=%v, kl=%v, ku=%v", path[1], path[2], a.Rows, m, kl, ku)
	} else if (notran && x.Rows < max(1, n)) || (tran && x.Rows < max(1, m)) {
		err = fmt.Errorf("(notran && x.Rows < max(1, n)) || (tran && x.Rows < max(1, m)): trans=%s, m=%v, n=%v, x.Rows=%v", trans, m, n, x.Rows)
	} else if (notran && b.Rows < max(1, m)) || (tran && b.Rows < max(1, n)) {
		err = fmt.Errorf("(notran && b.Rows < max(1, m)) || (tran && b.Rows < max(1, n)): trans=%s, m=%v, n=%v, b.Rows=%v", trans, m, n, b.Rows)
	}
	if err != nil {
		gltest.Xerbla2("Dlarhs", err)
		return
	}

	//     Initialize X to NRHS random vectors unless XTYPE = 'C'.
	if tran {
		nx = m
		mb = n
	} else {
		nx = n
		mb = m
	}
	if xtype != 'C' {
		for j = 1; j <= nrhs; j++ {
			golapack.Dlarnv(2, iseed, n, x.Off(0, j-1).Vector())
		}
	}

	//     Multiply X by op( A ) using an appropriate
	//     matrix multiply routine.
	if c2 == "ge" || c2 == "qr" || c2 == "lq" || c2 == "ql" || c2 == "rq" {
		//        General matrix
		err = b.Gemm(trans, mat.NoTrans, mb, nrhs, nx, one, a, x, zero)

	} else if c2 == "po" || c2 == "sy" {
		//        Symmetric matrix, 2-D storage
		err = b.Symm(mat.Left, uplo, n, nrhs, one, a, x, zero)

	} else if c2 == "gb" {
		//        General matrix, band storage
		for j = 1; j <= nrhs; j++ {
			err = b.Off(0, j-1).Vector().Gbmv(trans, mb, nx, kl, ku, one, a, x.Off(0, j-1).Vector(), 1, zero, 1)
		}

	} else if c2 == "pb" {
		//        Symmetric matrix, band storage
		for j = 1; j <= nrhs; j++ {
			err = b.Off(0, j-1).Vector().Sbmv(uplo, n, kl, one, a, x.Off(0, j-1).Vector(), 1, zero, 1)
		}

	} else if c2 == "pp" || c2 == "sp" {
		//        Symmetric matrix, packed storage
		for j = 1; j <= nrhs; j++ {
			err = b.Off(0, j-1).Vector().Spmv(uplo, n, one, a.OffIdx(0).Vector(), x.Off(0, j-1).Vector(), 1, zero, 1)
		}

	} else if c2 == "tr" {
		//        Triangular matrix.  Note that for triangular matrices,
		//           KU = 1 => non-unit triangular
		//           KU = 2 => unit triangular
		golapack.Dlacpy(Full, n, nrhs, x, b)
		if ku == 2 {
			diag = Unit
		} else {
			diag = NonUnit
		}
		err = b.Trmm(mat.Left, uplo, trans, diag, n, nrhs, one, a)

	} else if c2 == "tp" {
		//        Triangular matrix, packed storage
		golapack.Dlacpy(Full, n, nrhs, x, b)
		if ku == 2 {
			diag = Unit
		} else {
			diag = NonUnit
		}
		for j = 1; j <= nrhs; j++ {
			err = b.Off(0, j-1).Vector().Tpmv(uplo, trans, diag, n, a.OffIdx(0).Vector(), 1)
		}
		//
	} else if c2 == "tb" {
		//        Triangular matrix, banded storage
		golapack.Dlacpy(Full, n, nrhs, x, b)
		if ku == 2 {
			diag = Unit
		} else {
			diag = NonUnit
		}
		for j = 1; j <= nrhs; j++ {
			err = b.Off(0, j-1).Vector().Tbmv(uplo, trans, diag, n, kl, a, 1)
		}

	} else {
		//        If PATH is none of the above, return with an error code.
		err = fmt.Errorf("Path is invalid: %s", path)
		gltest.Xerbla2("Dlarhs", err)
	}

	return
}
