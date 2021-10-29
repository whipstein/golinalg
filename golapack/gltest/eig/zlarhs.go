package eig

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zlarhs chooses a set of NRHS random solution vectors and sets
// up the right hand sides for the linear system
//    op( A ) * X = B,
// where op( A ) may be A, A**T (transpose of A), or A**H (conjugate
// transpose of A).
func zlarhs(path string, xtype byte, uplo mat.MatUplo, trans mat.MatTrans, m, n, kl, ku, nrhs int, a, x, b *mat.CMatrix, iseed *[]int) (err error) {
	var band, gen, notran, qrs, sym, tran, tri bool
	var diag mat.MatDiag
	var one, zero complex128
	var j, mb, nx int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	c1 := path[0]
	c2 := path[1:3]
	tran = trans.IsTrans()
	notran = !tran
	gen = path[1] == 'g'
	qrs = path[1] == 'q' || path[2] == 'q'
	sym = path[1] == 'p' || path[1] == 's' || path[1] == 'h'
	tri = path[1] == 't'
	band = path[2] == 'b'
	if c1 != 'Z' {
		err = fmt.Errorf("c1 != 'Z': c1='%c'", c1)
	} else if !(xtype == 'N' || xtype == 'C') {
		err = fmt.Errorf("!(xtype == 'N' || xtype == 'C'): xtype='%c'", xtype)
	} else if (sym || tri) && !(uplo == Upper || uplo == Lower) {
		err = fmt.Errorf("(sym || tri) && !(uplo == Upper || uplo == Lower): sym=%v, tri=%v, uplo=%s", sym, tri, uplo)
	} else if (gen || qrs) && !(tran || trans == NoTrans) {
		err = fmt.Errorf("(gen || qrs) && !(tran || trans == NoTrans): gen=%v, qrs=%v, tran=%v, trans=%s", gen, qrs, tran, trans)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if band && kl < 0 {
		err = fmt.Errorf("band && kl < 0: band=%v, kl=%v", band, kl)
	} else if band && ku < 0 {
		err = fmt.Errorf("band && ku < 0: band=%v, ku=%v", band, ku)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if (!band && a.Rows < max(1, m)) || (band && (sym || tri) && a.Rows < kl+1) || (band && gen && a.Rows < kl+ku+1) {
		err = fmt.Errorf("(!band && a.Rows < max(1, m)) || (band && (sym || tri) && a.Rows < kl+1) || (band && gen && a.Rows < kl+ku+1): band=%v, sym=%v, tri=%v, gen=%v, a.Rows=%v, m=%v, kl=%v, ku=%v", band, sym, tri, gen, a.Rows, m, kl, ku)
	} else if (notran && x.Rows < max(1, n)) || (tran && x.Rows < max(1, m)) {
		err = fmt.Errorf("(notran && x.Rows < max(1, n)) || (tran && x.Rows < max(1, m)): notran=%v, tran=%v, x.Rows=%v, m=%v, n=%v", notran, tran, x.Rows, m, n)
	} else if (notran && b.Rows < max(1, m)) || (tran && b.Rows < max(1, n)) {
		err = fmt.Errorf("(notran && b.Rows < max(1, m)) || (tran && b.Rows < max(1, n)): notran=%v, tran=%v, b.Rows=%v, m=%v, n=%v", notran, tran, b.Rows, m, n)
	}
	if err != nil {
		gltest.Xerbla2("zlarhs", err)
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
			golapack.Zlarnv(2, iseed, n, x.CVector(0, j-1))
		}
	}

	//     Multiply X by op( A ) using an appropriate
	//     matrix multiply routine.
	if c2 == "ge" || c2 == "qr" || c2 == "lq" || c2 == "ql" || c2 == "rq" {
		//        General matrix
		if b.Rows < mb {
			if err = goblas.Zgemm(trans, NoTrans, mb, nrhs, nx, one, a, x, zero, b.Off(0, 0).UpdateRows(mb)); err != nil {
				panic(err)
			}
		} else {
			if err = goblas.Zgemm(trans, NoTrans, mb, nrhs, nx, one, a, x, zero, b); err != nil {
				panic(err)
			}
		}

	} else if c2 == "po" || c2 == "he" {
		//        Hermitian matrix, 2-D storage
		if err = goblas.Zhemm(Left, uplo, n, nrhs, one, a, x, zero, b); err != nil {
			panic(err)
		}

	} else if c2 == "sy" {
		//        Symmetric matrix, 2-D storage
		if err = goblas.Zsymm(Left, uplo, n, nrhs, one, a, x, zero, b); err != nil {
			panic(err)
		}

	} else if c2 == "gb" {
		//        General matrix, band storage
		for j = 1; j <= nrhs; j++ {
			if err = goblas.Zgbmv(trans, m, n, kl, ku, one, a, x.CVector(0, j-1, 1), zero, b.CVector(0, j-1, 1)); err != nil {
				panic(err)
			}
		}

	} else if c2 == "pb" || c2 == "hb" {
		//        Hermitian matrix, band storage
		for j = 1; j <= nrhs; j++ {
			if err = goblas.Zhbmv(uplo, n, kl, one, a, x.CVector(0, j-1, 1), zero, b.CVector(0, j-1, 1)); err != nil {
				panic(err)
			}
		}

	} else if c2 == "sb" {
		//        Symmetric matrix, band storage
		for j = 1; j <= nrhs; j++ {
			if err = zsbmv(uplo, n, kl, one, a, x.CVector(0, j-1), 1, zero, b.CVector(0, j-1), 1); err != nil {
				panic(err)
			}
		}

	} else if c2 == "pp" || c2 == "hp" {
		//        Hermitian matrix, packed storage
		for j = 1; j <= nrhs; j++ {
			if err = goblas.Zhpmv(uplo, n, one, a.CVector(0, 0), x.CVector(0, j-1, 1), zero, b.CVector(0, j-1, 1)); err != nil {
				panic(err)
			}
		}

	} else if c2 == "sp" {
		//        Symmetric matrix, packed storage
		for j = 1; j <= nrhs; j++ {
			if err = golapack.Zspmv(uplo, n, one, a.CVector(0, 0), x.CVector(0, j-1, 1), zero, b.CVector(0, j-1, 1)); err != nil {
				panic(err)
			}
		}

	} else if c2 == "tr" {
		//        Triangular matrix.  Note that for triangular matrices,
		//           KU = 1 => non-unit triangular
		//           KU = 2 => unit triangular
		golapack.Zlacpy(Full, n, nrhs, x, b)
		if ku == 2 {
			diag = 'U'
		} else {
			diag = 'N'
		}
		if err = goblas.Ztrmm(Left, uplo, trans, diag, n, nrhs, one, a, b); err != nil {
			panic(err)
		}

	} else if c2 == "tp" {
		//        Triangular matrix, packed storage
		golapack.Zlacpy(Full, n, nrhs, x, b)
		if ku == 2 {
			diag = 'U'
		} else {
			diag = 'N'
		}
		for j = 1; j <= nrhs; j++ {
			if err = goblas.Ztpmv(uplo, trans, diag, n, a.CVector(0, 0), b.CVector(0, j-1, 1)); err != nil {
				panic(err)
			}
		}

	} else if c2 == "tb" {
		//        Triangular matrix, banded storage
		golapack.Zlacpy(Full, n, nrhs, x, b)
		if ku == 2 {
			diag = 'U'
		} else {
			diag = 'N'
		}
		for j = 1; j <= nrhs; j++ {
			if err = goblas.Ztbmv(uplo, trans, diag, n, kl, a, b.CVector(0, j-1, 1)); err != nil {
				panic(err)
			}
		}

	} else {
		//        If none of the above, set INFO = -1 and return
		err = fmt.Errorf("-1")
		gltest.Xerbla2("zlarhs", err)
	}

	return
}
