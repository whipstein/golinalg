package golapack

import "math"

// Dlasdt creates a tree of subproblems for bidiagonal divide and
// conquer.
func Dlasdt(n int, inode, ndiml, ndimr *[]int, msub int) (lvl, nd int) {
	var temp, two float64
	var i, il, ir, llst, maxn, ncrnt, nlvl int

	two = 2.0

	//     Find the number of levels on the tree.
	maxn = max(1, n)
	temp = math.Log(float64(maxn)/float64(msub+1)) / math.Log(two)
	lvl = int(temp) + 1

	i = n / 2
	(*inode)[0] = i + 1
	(*ndiml)[0] = i
	(*ndimr)[0] = n - i - 1
	il = 0
	ir = 1
	llst = 1
	for nlvl = 1; nlvl <= lvl-1; nlvl++ {
		//        Constructing the tree at (NLVL+1)-st level. The number of
		//        nodes created on this level is LLST * 2.
		for i = 0; i <= llst-1; i++ {
			il = il + 2
			ir = ir + 2
			ncrnt = llst + i
			(*ndiml)[il-1] = (*ndiml)[ncrnt-1] / 2
			(*ndimr)[il-1] = (*ndiml)[ncrnt-1] - (*ndiml)[il-1] - 1
			(*inode)[il-1] = (*inode)[ncrnt-1] - (*ndimr)[il-1] - 1
			(*ndiml)[ir-1] = (*ndimr)[ncrnt-1] / 2
			(*ndimr)[ir-1] = (*ndimr)[ncrnt-1] - (*ndiml)[ir-1] - 1
			(*inode)[ir-1] = (*inode)[ncrnt-1] + (*ndiml)[ir-1] + 1
		}
		llst = llst * 2
	}
	nd = llst*2 - 1

	return
}
