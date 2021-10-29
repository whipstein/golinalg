package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlalsd uses the singular value decomposition of A to solve the least
// squares problem of finding X to minimize the Euclidean norm of each
// column of A*X-B, where A is N-by-N upper bidiagonal, and X and B
// are N-by-NRHS. The solution X overwrites B.
//
// The singular values of A smaller than RCOND times the largest
// singular value are treated as zero in solving the least squares
// problem; in this case a minimum norm solution is returned.
// The actual singular values are returned in D in ascending order.
//
// This code makes very mild assumptions about floating point
// arithmetic. It will work on machines with a guard digit in
// add/subtract, or on those binary machines without guard digits
// which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2.
// It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Zlalsd(uplo mat.MatUplo, smlsiz, n, nrhs int, d, e *mat.Vector, b *mat.CMatrix, rcond float64, work *mat.CVector, rwork *mat.Vector, iwork *[]int) (rank, info int, err error) {
	var czero complex128
	var cs, eps, one, orgnrm, r, rcnd, sn, tol, two, zero float64
	var bx, bxst, c, difl, difr, givcol, givnum, givptr, i, icmpq1, icmpq2, irwb, irwib, irwrb, irwu, irwvt, irwwrk, iwk, j, jcol, jimag, jreal, jrow, k, nlvl, nm1, nrwork, nsize, nsub, perm, poles, s, sizei, smlszp, sqre, st, st1, u, vt, z int

	zero = 0.0
	one = 1.0
	two = 2.0
	czero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 1 {
		err = fmt.Errorf("nrhs < 1: nrhs=%v", nrhs)
	} else if (b.Rows < 1) || (b.Rows < n) {
		err = fmt.Errorf("(b.Rows < 1) || (b.Rows < n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zlalsd", err)
		return
	}

	eps = Dlamch(Epsilon)

	//     Set up the tolerance.
	if (rcond <= zero) || (rcond >= one) {
		rcnd = eps
	} else {
		rcnd = rcond
	}

	rank = 0

	//     Quick return if possible.
	if n == 0 {
		return
	} else if n == 1 {
		if d.Get(0) == zero {
			Zlaset(Full, 1, nrhs, czero, czero, b)
		} else {
			rank = 1
			if err = Zlascl('G', 0, 0, d.Get(0), one, 1, nrhs, b); err != nil {
				panic(err)
			}
			d.Set(0, d.GetMag(0))
		}
		return
	}

	//     Rotate the matrix if it is lower bidiagonal.
	if uplo == Lower {
		for i = 1; i <= n-1; i++ {
			cs, sn, r = Dlartg(d.Get(i-1), e.Get(i-1))
			d.Set(i-1, r)
			e.Set(i-1, sn*d.Get(i))
			d.Set(i, cs*d.Get(i))
			if nrhs == 1 {
				goblas.Zdrot(1, b.CVector(i-1, 0, 1), b.CVector(i, 0, 1), cs, sn)
			} else {
				rwork.Set(i*2-1-1, cs)
				rwork.Set(i*2-1, sn)
			}
		}
		if nrhs > 1 {
			for i = 1; i <= nrhs; i++ {
				for j = 1; j <= n-1; j++ {
					cs = rwork.Get(j*2 - 1 - 1)
					sn = rwork.Get(j*2 - 1)
					goblas.Zdrot(1, b.CVector(j-1, i-1, 1), b.CVector(j, i-1, 1), cs, sn)
				}
			}
		}
	}

	//     Scale.
	nm1 = n - 1
	orgnrm = Dlanst('M', n, d, e)
	if orgnrm == zero {
		Zlaset(Full, n, nrhs, czero, czero, b)
		return
	}

	if err = Dlascl('G', 0, 0, orgnrm, one, n, 1, d.Matrix(n, opts)); err != nil {
		panic(err)
	}
	if err = Dlascl('G', 0, 0, orgnrm, one, nm1, 1, e.Matrix(nm1, opts)); err != nil {
		panic(err)
	}

	//     If N is smaller than the minimum divide size SMLSIZ, then solve
	//     the problem with another solver.
	if n <= smlsiz {
		irwu = 1
		irwvt = irwu + n*n
		irwwrk = irwvt + n*n
		irwrb = irwwrk
		irwib = irwrb + n*nrhs
		irwb = irwib + n*nrhs
		Dlaset(Full, n, n, zero, one, rwork.MatrixOff(irwu-1, n, opts))
		Dlaset(Full, n, n, zero, one, rwork.MatrixOff(irwvt-1, n, opts))
		if info, err = Dlasdq(Upper, 0, n, n, n, 0, d, e, rwork.MatrixOff(irwvt-1, n, opts), rwork.MatrixOff(irwu-1, n, opts), rwork.MatrixOff(irwwrk-1, 1, opts), rwork.Off(irwwrk-1)); err != nil {
			panic(err)
		}
		if info != 0 {
			return
		}

		//        In the real version, B is passed to DLASDQ and multiplied
		//        internally by Q**H. Here B is complex and that product is
		//        computed below in two steps (real and imaginary parts).
		j = irwb - 1
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = 1; jrow <= n; jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetRe(jrow-1, jcol-1))
			}
		}
		if err = goblas.Dgemm(Trans, NoTrans, n, nrhs, n, one, rwork.MatrixOff(irwu-1, n, opts), rwork.MatrixOff(irwb-1, n, opts), zero, rwork.MatrixOff(irwrb-1, n, opts)); err != nil {
			panic(err)
		}
		j = irwb - 1
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = 1; jrow <= n; jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetIm(jrow-1, jcol-1))
			}
		}
		if err = goblas.Dgemm(Trans, NoTrans, n, nrhs, n, one, rwork.MatrixOff(irwu-1, n, opts), rwork.MatrixOff(irwb-1, n, opts), zero, rwork.MatrixOff(irwib-1, n, opts)); err != nil {
			panic(err)
		}
		jreal = irwrb - 1
		jimag = irwib - 1
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = 1; jrow <= n; jrow++ {
				jreal = jreal + 1
				jimag = jimag + 1
				b.Set(jrow-1, jcol-1, complex(rwork.Get(jreal-1), rwork.Get(jimag-1)))
			}
		}

		tol = rcnd * d.GetMag(goblas.Idamax(n, d.Off(0, 1))-1)
		for i = 1; i <= n; i++ {
			if d.Get(i-1) <= tol {
				Zlaset(Full, 1, nrhs, czero, czero, b.Off(i-1, 0))
			} else {
				if err = Zlascl('G', 0, 0, d.Get(i-1), one, 1, nrhs, b.Off(i-1, 0)); err != nil {
					panic(err)
				}
				rank = rank + 1
			}
		}

		//        Since B is complex, the following call to DGEMM is performed
		//        in two steps (real and imaginary parts). That is for V * B
		//        (in the real version of the code V**H is stored in WORK).
		//
		//        CALL DGEMM( 'T', 'N', N, NRHS, N, ONE, WORK, N, B, LDB, ZERO,
		//    $               WORK( NWORK ), N )
		j = irwb - 1
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = 1; jrow <= n; jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetRe(jrow-1, jcol-1))
			}
		}
		if err = goblas.Dgemm(Trans, NoTrans, n, nrhs, n, one, rwork.MatrixOff(irwvt-1, n, opts), rwork.MatrixOff(irwb-1, n, opts), zero, rwork.MatrixOff(irwrb-1, n, opts)); err != nil {
			panic(err)
		}
		j = irwb - 1
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = 1; jrow <= n; jrow++ {
				j = j + 1
				rwork.Set(j-1, b.GetIm(jrow-1, jcol-1))
			}
		}
		if err = goblas.Dgemm(Trans, NoTrans, n, nrhs, n, one, rwork.MatrixOff(irwvt-1, n, opts), rwork.MatrixOff(irwb-1, n, opts), zero, rwork.MatrixOff(irwib-1, n, opts)); err != nil {
			panic(err)
		}
		jreal = irwrb - 1
		jimag = irwib - 1
		for jcol = 1; jcol <= nrhs; jcol++ {
			for jrow = 1; jrow <= n; jrow++ {
				jreal = jreal + 1
				jimag = jimag + 1
				b.Set(jrow-1, jcol-1, complex(rwork.Get(jreal-1), rwork.Get(jimag-1)))
			}
		}

		//        Unscale.
		if err = Dlascl('G', 0, 0, one, orgnrm, n, 1, d.Matrix(n, opts)); err != nil {
			panic(err)
		}
		if err = Dlasrt('D', n, d); err != nil {
			panic(err)
		}
		if err = Zlascl('G', 0, 0, orgnrm, one, n, nrhs, b); err != nil {
			panic(err)
		}

		return
	}

	//     Book-keeping and setting up some constants.
	nlvl = int(math.Log(float64(n)/float64(smlsiz+1))/math.Log(two)) + 1

	smlszp = smlsiz + 1

	u = 1
	vt = 1 + smlsiz*n
	difl = vt + smlszp*n
	difr = difl + nlvl*n
	z = difr + nlvl*n*2
	c = z + nlvl*n
	s = c + n
	poles = s + n
	givnum = poles + 2*nlvl*n
	nrwork = givnum + 2*nlvl*n
	bx = 1

	irwrb = nrwork
	irwib = irwrb + smlsiz*nrhs
	irwb = irwib + smlsiz*nrhs

	sizei = 1 + n
	k = sizei + n
	givptr = k + n
	perm = givptr + n
	givcol = perm + nlvl*n
	iwk = givcol + nlvl*n*2

	st = 1
	sqre = 0
	icmpq1 = 1
	icmpq2 = 0
	nsub = 0

	for i = 1; i <= n; i++ {
		if d.GetMag(i-1) < eps {
			d.Set(i-1, math.Copysign(eps, d.Get(i-1)))
		}
	}

	for i = 1; i <= nm1; i++ {
		if (e.GetMag(i-1) < eps) || (i == nm1) {
			nsub = nsub + 1
			(*iwork)[nsub-1] = st

			//           Subproblem found. First determine its size and then
			//           apply divide and conquer on it.
			if i < nm1 {
				//              A subproblem with E(I) small for I < NM1.
				nsize = i - st + 1
				(*iwork)[sizei+nsub-1-1] = nsize
			} else if e.GetMag(i-1) >= eps {
				//              A subproblem with E(NM1) not too small but I = NM1.
				nsize = n - st + 1
				(*iwork)[sizei+nsub-1-1] = nsize
			} else {
				//              A subproblem with E(NM1) small. This implies an
				//              1-by-1 subproblem at D(N), which is not solved
				//              explicitly.
				nsize = i - st + 1
				(*iwork)[sizei+nsub-1-1] = nsize
				nsub = nsub + 1
				(*iwork)[nsub-1] = n
				(*iwork)[sizei+nsub-1-1] = 1
				goblas.Zcopy(nrhs, b.CVector(n-1, 0, *&b.Rows), work.Off(bx+nm1-1, n))
			}
			st1 = st - 1
			if nsize == 1 {
				//              This is a 1-by-1 subproblem and is not solved
				//              explicitly.
				goblas.Zcopy(nrhs, b.CVector(st-1, 0, *&b.Rows), work.Off(bx+st1-1, n))
			} else if nsize <= smlsiz {
				//              This is a small subproblem and is solved by DLASDQ.
				Dlaset(Full, nsize, nsize, zero, one, rwork.MatrixOff(vt+st1-1, n, opts))
				Dlaset(Full, nsize, nsize, zero, one, rwork.MatrixOff(u+st1-1, n, opts))
				if info, err = Dlasdq(Upper, 0, nsize, nsize, nsize, 0, d.Off(st-1), e.Off(st-1), rwork.MatrixOff(vt+st1-1, n, opts), rwork.MatrixOff(u+st1-1, n, opts), rwork.MatrixOff(nrwork-1, 1, opts), rwork.Off(nrwork-1)); err != nil {
					panic(err)
				}
				if info != 0 {
					return
				}

				//              In the real version, B is passed to DLASDQ and multiplied
				//              internally by Q**H. Here B is complex and that product is
				//              computed below in two steps (real and imaginary parts).
				j = irwb - 1
				for jcol = 1; jcol <= nrhs; jcol++ {
					for jrow = st; jrow <= st+nsize-1; jrow++ {
						j = j + 1
						rwork.Set(j-1, b.GetRe(jrow-1, jcol-1))
					}
				}
				if err = goblas.Dgemm(Trans, NoTrans, nsize, nrhs, nsize, one, rwork.MatrixOff(u+st1-1, n, opts), rwork.MatrixOff(irwb-1, nsize, opts), zero, rwork.MatrixOff(irwrb-1, nsize, opts)); err != nil {
					panic(err)
				}
				j = irwb - 1
				for jcol = 1; jcol <= nrhs; jcol++ {
					for jrow = st; jrow <= st+nsize-1; jrow++ {
						j = j + 1
						rwork.Set(j-1, b.GetIm(jrow-1, jcol-1))
					}
				}
				if err = goblas.Dgemm(Trans, NoTrans, nsize, nrhs, nsize, one, rwork.MatrixOff(u+st1-1, n, opts), rwork.MatrixOff(irwb-1, nsize, opts), zero, rwork.MatrixOff(irwib-1, nsize, opts)); err != nil {
					panic(err)
				}
				jreal = irwrb - 1
				jimag = irwib - 1
				for jcol = 1; jcol <= nrhs; jcol++ {
					for jrow = st; jrow <= st+nsize-1; jrow++ {
						jreal = jreal + 1
						jimag = jimag + 1
						b.Set(jrow-1, jcol-1, complex(rwork.Get(jreal-1), rwork.Get(jimag-1)))
					}
				}

				Zlacpy(Full, nsize, nrhs, b.Off(st-1, 0), work.CMatrixOff(bx+st1-1, n, opts))
			} else {
				//              A large problem. Solve it using divide and conquer.
				if info, err = Dlasda(icmpq1, smlsiz, nsize, sqre, d.Off(st-1), e.Off(st-1), rwork.MatrixOff(u+st1-1, n, opts), rwork.MatrixOff(vt+st1-1, n, opts), toSlice(iwork, k+st1-1), rwork.MatrixOff(difl+st1-1, n, opts), rwork.MatrixOff(difr+st1-1, n, opts), rwork.MatrixOff(z+st1-1, n, opts), rwork.MatrixOff(poles+st1-1, n, opts), toSlice(iwork, givptr+st1-1), toSlice(iwork, givcol+st1-1), n, toSlice(iwork, perm+st1-1), rwork.MatrixOff(givnum+st1-1, n, opts), rwork.Off(c+st1-1), rwork.Off(s+st1-1), rwork.Off(nrwork-1), toSlice(iwork, iwk-1)); err != nil {
					panic(err)
				}
				if info != 0 {
					return
				}
				bxst = bx + st1
				if err = Zlalsa(icmpq2, smlsiz, nsize, nrhs, b.Off(st-1, 0), work.CMatrixOff(bxst-1, n, opts), rwork.MatrixOff(u+st1-1, n, opts), rwork.MatrixOff(vt+st1-1, n, opts), toSlice(iwork, k+st1-1), rwork.MatrixOff(difl+st1-1, n, opts), rwork.MatrixOff(difr+st1-1, n, opts), rwork.MatrixOff(z+st1-1, n, opts), rwork.MatrixOff(poles+st1-1, n, opts), toSlice(iwork, givptr+st1-1), toSlice(iwork, givcol+st1-1), n, toSlice(iwork, perm+st1-1), rwork.MatrixOff(givnum+st1-1, n, opts), rwork.Off(c+st1-1), rwork.Off(s+st1-1), rwork.Off(nrwork-1), toSlice(iwork, iwk-1)); err != nil {
					panic(err)
				}
				if info != 0 {
					return
				}
			}
			st = i + 1
		}
	}

	//     Apply the singular values and treat the tiny ones as zero.
	tol = rcnd * d.GetMag(goblas.Idamax(n, d.Off(0, 1))-1)

	for i = 1; i <= n; i++ {
		//        Some of the elements in D can be negative because 1-by-1
		//        subproblems were not solved explicitly.
		if d.GetMag(i-1) <= tol {
			Zlaset(Full, 1, nrhs, czero, czero, work.CMatrixOff(bx+i-1-1, n, opts))
		} else {
			rank = rank + 1
			if err = Zlascl('G', 0, 0, d.Get(i-1), one, 1, nrhs, work.CMatrixOff(bx+i-1-1, n, opts)); err != nil {
				panic(err)
			}
		}
		d.Set(i-1, d.GetMag(i-1))
	}

	//     Now apply back the right singular vectors.
	icmpq2 = 1
	for i = 1; i <= nsub; i++ {
		st = (*iwork)[i-1]
		st1 = st - 1
		nsize = (*iwork)[sizei+i-1-1]
		bxst = bx + st1
		if nsize == 1 {
			goblas.Zcopy(nrhs, work.Off(bxst-1, n), b.CVector(st-1, 0, *&b.Rows))
		} else if nsize <= smlsiz {
			//           Since B and BX are complex, the following call to DGEMM
			//           is performed in two steps (real and imaginary parts).
			//
			//           CALL DGEMM( 'T', 'N', NSIZE, NRHS, NSIZE, ONE,
			//    $                  RWORK( VT+ST1 ), N, RWORK( BXST ), N, ZERO,
			//    $                  B( ST, 1 ), LDB )
			j = bxst - n - 1
			jreal = irwb - 1
			for jcol = 1; jcol <= nrhs; jcol++ {
				j = j + n
				for jrow = 1; jrow <= nsize; jrow++ {
					jreal = jreal + 1
					rwork.Set(jreal-1, work.GetRe(j+jrow-1))
				}
			}
			if err = goblas.Dgemm(Trans, NoTrans, nsize, nrhs, nsize, one, rwork.MatrixOff(vt+st1-1, n, opts), rwork.MatrixOff(irwb-1, nsize, opts), zero, rwork.MatrixOff(irwrb-1, nsize, opts)); err != nil {
				panic(err)
			}
			j = bxst - n - 1
			jimag = irwb - 1
			for jcol = 1; jcol <= nrhs; jcol++ {
				j = j + n
				for jrow = 1; jrow <= nsize; jrow++ {
					jimag = jimag + 1
					rwork.Set(jimag-1, work.GetIm(j+jrow-1))
				}
			}
			if err = goblas.Dgemm(Trans, NoTrans, nsize, nrhs, nsize, one, rwork.MatrixOff(vt+st1-1, n, opts), rwork.MatrixOff(irwb-1, nsize, opts), zero, rwork.MatrixOff(irwib-1, nsize, opts)); err != nil {
				panic(err)
			}
			jreal = irwrb - 1
			jimag = irwib - 1
			for jcol = 1; jcol <= nrhs; jcol++ {
				for jrow = st; jrow <= st+nsize-1; jrow++ {
					jreal = jreal + 1
					jimag = jimag + 1
					b.Set(jrow-1, jcol-1, complex(rwork.Get(jreal-1), rwork.Get(jimag-1)))
				}
			}
		} else {
			if err = Zlalsa(icmpq2, smlsiz, nsize, nrhs, work.CMatrixOff(bxst-1, n, opts), b.Off(st-1, 0), rwork.MatrixOff(u+st1-1, n, opts), rwork.MatrixOff(vt+st1-1, n, opts), toSlice(iwork, k+st1-1), rwork.MatrixOff(difl+st1-1, n, opts), rwork.MatrixOff(difr+st1-1, n, opts), rwork.MatrixOff(z+st1-1, n, opts), rwork.MatrixOff(poles+st1-1, n, opts), toSlice(iwork, givptr+st1-1), toSlice(iwork, givcol+st1-1), n, toSlice(iwork, perm+st1-1), rwork.MatrixOff(givnum+st1-1, n, opts), rwork.Off(c+st1-1), rwork.Off(s+st1-1), rwork.Off(nrwork-1), toSlice(iwork, iwk-1)); err != nil {
				panic(err)
			}
			if info != 0 {
				return
			}
		}
	}

	//     Unscale and sort the singular values.
	if err = Dlascl('G', 0, 0, one, orgnrm, n, 1, d.Matrix(n, opts)); err != nil {
		panic(err)
	}
	if err = Dlasrt('D', n, d); err != nil {
		panic(err)
	}
	if err = Zlascl('G', 0, 0, orgnrm, one, n, nrhs, b); err != nil {
		panic(err)
	}

	return
}
