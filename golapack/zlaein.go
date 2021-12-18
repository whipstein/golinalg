package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Zlaein uses inverse iteration to find a right or left eigenvector
// corresponding to the eigenvalue W of a complex upper Hessenberg
// matrix H.
func Zlaein(rightv, noinit bool, n int, h *mat.CMatrix, w complex128, v *mat.CVector, b *mat.CMatrix, rwork *mat.Vector, eps3, smlnum float64) (info int) {
	var normin byte
	var trans mat.MatTrans
	var ei, ej, temp, x, zero complex128
	var growto, nrmsml, one, rootn, rtemp, scale, tenth, vnorm float64
	var i, its, j int
	var err error

	one = 1.0
	tenth = 1.0e-1
	zero = (0.0 + 0.0*1i)

	//     GROWTO is the threshold used in the acceptance test for an
	//     eigenvector.
	rootn = math.Sqrt(float64(n))
	growto = tenth / rootn
	nrmsml = math.Max(one, eps3*rootn) * smlnum

	//     Form B = H - W*I (except that the subdiagonal elements are not
	//     stored).
	for j = 1; j <= n; j++ {
		for i = 1; i <= j-1; i++ {
			b.Set(i-1, j-1, h.Get(i-1, j-1))
		}
		b.Set(j-1, j-1, h.Get(j-1, j-1)-w)
	}

	if noinit {
		//        Initialize V.
		for i = 1; i <= n; i++ {
			v.SetRe(i-1, eps3)
		}
	} else {
		//        Scale supplied initial vector.
		vnorm = v.Nrm2(n, 1)
		v.Dscal(n, (eps3*rootn)/math.Max(vnorm, nrmsml), 1)
	}

	if rightv {
		//        LU decomposition with partial pivoting of B, replacing zero
		//        pivots by EPS3.
		for i = 1; i <= n-1; i++ {
			ei = h.Get(i, i-1)
			if cabs1(b.Get(i-1, i-1)) < cabs1(ei) {
				//              Interchange rows and eliminate.
				x = Zladiv(b.Get(i-1, i-1), ei)
				b.Set(i-1, i-1, ei)
				for j = i + 1; j <= n; j++ {
					temp = b.Get(i, j-1)
					b.Set(i, j-1, b.Get(i-1, j-1)-x*temp)
					b.Set(i-1, j-1, temp)
				}
			} else {
				//              Eliminate without interchange.
				if b.Get(i-1, i-1) == zero {
					b.SetRe(i-1, i-1, eps3)
				}
				x = Zladiv(ei, b.Get(i-1, i-1))
				if x != zero {
					for j = i + 1; j <= n; j++ {
						b.Set(i, j-1, b.Get(i, j-1)-x*b.Get(i-1, j-1))
					}
				}
			}
		}
		if b.Get(n-1, n-1) == zero {
			b.SetRe(n-1, n-1, eps3)
		}

		trans = NoTrans

	} else {
		//        UL decomposition with partial pivoting of B, replacing zero
		//        pivots by EPS3.
		for j = n; j >= 2; j-- {
			ej = h.Get(j-1, j-1-1)
			if cabs1(b.Get(j-1, j-1)) < cabs1(ej) {
				//              Interchange columns and eliminate.
				x = Zladiv(b.Get(j-1, j-1), ej)
				b.Set(j-1, j-1, ej)
				for i = 1; i <= j-1; i++ {
					temp = b.Get(i-1, j-1-1)
					b.Set(i-1, j-1-1, b.Get(i-1, j-1)-x*temp)
					b.Set(i-1, j-1, temp)
				}
			} else {
				//              Eliminate without interchange.
				if b.Get(j-1, j-1) == zero {
					b.SetRe(j-1, j-1, eps3)
				}
				x = Zladiv(ej, b.Get(j-1, j-1))
				if x != zero {
					for i = 1; i <= j-1; i++ {
						b.Set(i-1, j-1-1, b.Get(i-1, j-1-1)-x*b.Get(i-1, j-1))
					}
				}
			}
		}
		if b.Get(0, 0) == zero {
			b.SetRe(0, 0, eps3)
		}

		trans = ConjTrans

	}

	normin = 'N'
	for its = 1; its <= n; its++ {
		//        Solve U*x = scale*v for a right eigenvector
		//          or U**H *x = scale*v for a left eigenvector,
		//        overwriting x on v.
		if scale, err = Zlatrs(Upper, trans, NonUnit, normin, n, b, v, rwork); err != nil {
			panic(err)
		}
		normin = 'Y'

		//        Test for sufficient growth in the norm of v.
		vnorm = v.Asum(n, 1)
		if vnorm >= growto*scale {
			goto label120
		}

		//        Choose new orthogonal starting vector and try again.
		rtemp = eps3 / (rootn + one)
		v.SetRe(0, eps3)
		for i = 2; i <= n; i++ {
			v.SetRe(i-1, rtemp)
		}
		v.Set(n-its, v.Get(n-its)-complex(eps3*rootn, 0))
	}

	//     Failure to find eigenvector in N iterations.
	info = 1

label120:
	;

	//     Normalize eigenvector.
	i = v.Iamax(n, 1)
	v.Dscal(n, one/cabs1(v.Get(i-1)), 1)

	return
}
