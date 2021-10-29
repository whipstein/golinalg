package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DsytrdSy2sb reduces a real symmetric matrix A to real symmetric
// band-diagonal form AB by a orthogonal similarity transformation:
// Q**T * A * Q = AB.
func DsytrdSy2sb(uplo mat.MatUplo, n, kd int, a, ab *mat.Matrix, tau, work *mat.Vector, lwork int) (err error) {
	var lquery, upper bool
	var half, one, rone, zero float64
	var i, j, lds1, lds2, ldt, ldw, lk, ls1, ls2, lt, lw, lwmin, pk, pn, s1pos, s2pos, tpos, wpos int

	rone = 1.0
	zero = 0.0
	one = 1.0
	half = 0.5

	//     Determine the minimal workspace size required
	//     and test the input parameters
	upper = uplo == Upper
	lquery = (lwork == -1)
	lwmin = Ilaenv2stage(4, "DsytrdSy2sb", []byte{' '}, n, kd, -1, -1)
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if ab.Rows < max(1, kd+1) {
		err = fmt.Errorf("ab.Rows < max(1, kd+1): ab.Rows=%v, kd=%v", ab.Rows, kd)
	} else if lwork < lwmin && !lquery {
		err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
	}

	if err != nil {
		gltest.Xerbla2("DsytrdSy2sb", err)
		return
	} else if lquery {
		work.Set(0, float64(lwmin))
		return
	}

	//     Quick return if possible
	//     Copy the upper/lower portion of A into AB
	if n <= kd+1 {
		if upper {
			for i = 1; i <= n; i++ {
				lk = min(kd+1, i)
				goblas.Dcopy(lk, a.Vector(i-lk, i-1, 1), ab.Vector(kd+1-lk, i-1, 1))
			}
		} else {
			for i = 1; i <= n; i++ {
				lk = min(kd+1, n-i+1)
				goblas.Dcopy(lk, a.Vector(i-1, i-1, 1), ab.Vector(0, i-1, 1))
			}
		}
		work.Set(0, 1)
		return
	}

	//     Determine the pointer position for the workspace
	ldt = kd
	lds1 = kd
	lt = ldt * kd
	lw = n * kd
	ls1 = lds1 * kd
	ls2 = lwmin - lt - lw - ls1
	//      LS2 = N*MAX(KD,FACTOPTNB)
	tpos = 1
	wpos = tpos + lt
	s1pos = wpos + lw
	s2pos = s1pos + ls1
	if upper {
		ldw = kd
		lds2 = kd
	} else {
		ldw = n
		lds2 = n
	}

	//     Set the workspace of the triangular matrix T to zero once such a
	//     way every time T is generated the upper/lower portion will be always zero
	Dlaset(Full, ldt, kd, zero, zero, work.MatrixOff(tpos-1, ldt, opts))

	if upper {
		for i = 1; i <= n-kd; i += kd {
			pn = n - i - kd + 1
			pk = min(n-i-kd+1, kd)

			//            Compute the LQ factorization of the current block
			if err = Dgelqf(kd, pn, a.Off(i-1, i+kd-1), tau.Off(i-1), work.Off(s2pos-1), ls2); err != nil {
				panic(err)
			}

			//            Copy the upper portion of A into AB
			for j = i; j <= i+pk-1; j++ {
				lk = min(kd, n-j) + 1
				goblas.Dcopy(lk, a.Vector(j-1, j-1), ab.Vector(kd, j-1, ab.Rows-1))
			}

			Dlaset(Lower, pk, pk, zero, one, a.Off(i-1, i+kd-1))

			//            Form the matrix T
			Dlarft('F', 'R', pn, pk, a.Off(i-1, i+kd-1), tau.Off(i-1), work.MatrixOff(tpos-1, ldt, opts))

			//            Compute W:
			if err = goblas.Dgemm(ConjTrans, NoTrans, pk, pn, pk, one, work.MatrixOff(tpos-1, ldt, opts), a.Off(i-1, i+kd-1), zero, work.MatrixOff(s2pos-1, lds2, opts)); err != nil {
				panic(err)
			}

			if err = goblas.Dsymm(Right, uplo, pk, pn, one, a.Off(i+kd-1, i+kd-1), work.MatrixOff(s2pos-1, lds2, opts), zero, work.MatrixOff(wpos-1, ldw, opts)); err != nil {
				panic(err)
			}

			if err = goblas.Dgemm(NoTrans, ConjTrans, pk, pk, pn, one, work.MatrixOff(wpos-1, ldw, opts), work.MatrixOff(s2pos-1, lds2, opts), zero, work.MatrixOff(s1pos-1, lds1, opts)); err != nil {
				panic(err)
			}

			if err = goblas.Dgemm(NoTrans, NoTrans, pk, pn, pk, half, work.MatrixOff(s1pos-1, lds1, opts), a.Off(i-1, i+kd-1), one, work.MatrixOff(wpos-1, ldw, opts)); err != nil {
				panic(err)
			}

			//            Update the unreduced submatrix A(i+kd:n,i+kd:n), using
			//            an update of the form:  A := A - V'*W - W'*V
			if err = goblas.Dsyr2k(uplo, ConjTrans, pn, pk, one, a.Off(i-1, i+kd-1), work.MatrixOff(wpos-1, ldw, opts), rone, a.Off(i+kd-1, i+kd-1)); err != nil {
				panic(err)
			}
		}

		//        Copy the upper band to AB which is the band storage matrix
		for j = n - kd + 1; j <= n; j++ {
			lk = min(kd, n-j) + 1
			goblas.Dcopy(lk, a.Vector(j-1, j-1), ab.Vector(kd, j-1, ab.Rows-1))
		}

	} else {
		//         Reduce the lower triangle of A to lower band matrix
		for i = 1; i <= n-kd; i += kd {
			pn = n - i - kd + 1
			pk = min(n-i-kd+1, kd)

			//            Compute the QR factorization of the current block
			if err = Dgeqrf(pn, kd, a.Off(i+kd-1, i-1), tau.Off(i-1), work.Off(s2pos-1), ls2); err != nil {
				panic(err)
			}

			//            Copy the upper portion of A into AB
			for j = i; j <= i+pk-1; j++ {
				lk = min(kd, n-j) + 1
				goblas.Dcopy(lk, a.Vector(j-1, j-1, 1), ab.Vector(0, j-1, 1))
			}

			Dlaset(Upper, pk, pk, zero, one, a.Off(i+kd-1, i-1))

			//            Form the matrix T
			Dlarft('F', 'C', pn, pk, a.Off(i+kd-1, i-1), tau.Off(i-1), work.MatrixOff(tpos-1, ldt, opts))

			//            Compute W:
			if err = goblas.Dgemm(NoTrans, NoTrans, pn, pk, pk, one, a.Off(i+kd-1, i-1), work.MatrixOff(tpos-1, ldt, opts), zero, work.MatrixOff(s2pos-1, lds2, opts)); err != nil {
				panic(err)
			}

			if err = goblas.Dsymm(Left, uplo, pn, pk, one, a.Off(i+kd-1, i+kd-1), work.MatrixOff(s2pos-1, lds2, opts), zero, work.MatrixOff(wpos-1, ldw, opts)); err != nil {
				panic(err)
			}

			if err = goblas.Dgemm(ConjTrans, NoTrans, pk, pk, pn, one, work.MatrixOff(s2pos-1, lds2, opts), work.MatrixOff(wpos-1, ldw, opts), zero, work.MatrixOff(s1pos-1, lds1, opts)); err != nil {
				panic(err)
			}

			if err = goblas.Dgemm(NoTrans, NoTrans, pn, pk, pk, half, a.Off(i+kd-1, i-1), work.MatrixOff(s1pos-1, lds1, opts), one, work.MatrixOff(wpos-1, ldw, opts)); err != nil {
				panic(err)
			}

			//            Update the unreduced submatrix A(i+kd:n,i+kd:n), using
			//            an update of the form:  A := A - V*W' - W*V'
			if err = goblas.Dsyr2k(uplo, NoTrans, pn, pk, one, a.Off(i+kd-1, i-1), work.MatrixOff(wpos-1, ldw, opts), rone, a.Off(i+kd-1, i+kd-1)); err != nil {
				panic(err)
			}
			//            ==================================================================
			//            RESTORE A FOR COMPARISON AND CHECKING TO BE REMOVED
			//             DO 45 J = I, I+PK-1
			//                LK = MIN( KD, N-J ) + 1
			//                CALL DCOPY( LK, AB( 1, J ), 1, A( J, J ), 1 )
			//   45        CONTINUE
			//            ==================================================================
		}

		//        Copy the lower band to AB which is the band storage matrix
		for j = n - kd + 1; j <= n; j++ {
			lk = min(kd, n-j) + 1
			goblas.Dcopy(lk, a.Vector(j-1, j-1, 1), ab.Vector(0, j-1, 1))
		}
	}

	work.Set(0, float64(lwmin))

	return
}
